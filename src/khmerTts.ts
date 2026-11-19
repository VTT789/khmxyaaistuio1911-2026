import path from 'path';
import os from 'os';
import fs from 'fs';
import crypto from 'crypto';
import { execFile } from 'child_process';
import { promisify } from 'util';

const execFileAsync = promisify(execFile);

export interface TtsSynthesisResult {
  buffer: Buffer;
  contentType: string;
}

export const KHMER_EXTRA_VOICES = [
  'km-KH-MMS-Neural',
  'km-KH-gTTS',
] as const;

const CHROME_USER_AGENT =
  'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

/**
 * Split long text into chunks under maxLen characters on natural boundaries
 * so Google Translate TTS does not truncate or reject long sentences.
 */
function splitTextForGtts(text: string, maxLen = 180): string[] {
  const cleaned = text.replace(/[\u200B-\u200D\uFEFF]/g, '').trim();
  if (!cleaned) return [];
  if (cleaned.length <= maxLen) return [cleaned];

  const chunks: string[] = [];
  let remaining = cleaned;

  while (remaining.length > maxLen) {
    let splitIdx = -1;
    const slice = remaining.slice(0, maxLen);
    for (const sep of ['។', '៕', '.', '!', '?', ';', ',', '/', ' ']) {
      const idx = slice.lastIndexOf(sep);
      if (idx > Math.floor(maxLen * 0.3)) {
        splitIdx = idx + 1;
        break;
      }
    }
    if (splitIdx === -1) {
      splitIdx = maxLen;
    }
    const piece = remaining.slice(0, splitIdx).trim();
    if (piece) chunks.push(piece);
    remaining = remaining.slice(splitIdx).trim();
  }
  if (remaining) chunks.push(remaining);
  return chunks;
}

/**
 * Convert an Edge-style rate string (e.g. "-25%", "+30%", "+0%") into an
 * ffmpeg atempo filter chain preserving pitch. Returns null if tempo === 1.0.
 */
function buildAtempoFilter(rateStr: string): string | null {
  const num = parseFloat(String(rateStr).replace(/%/g, '').trim());
  if (isNaN(num) || num === 0) return null;

  // Map -100%..+100% to 0.25x..2.0x speed
  const rawTempo = 1 + num / 100;
  const tempo = Math.max(0.25, Math.min(2.0, rawTempo));
  if (Math.abs(tempo - 1.0) < 0.01) return null;

  if (tempo < 0.5) {
    const second = Math.max(0.5, tempo / 0.5);
    return `atempo=0.5,atempo=${second.toFixed(3)}`;
  }
  return `atempo=${tempo.toFixed(3)}`;
}

async function applyAudioTempo(
  buffer: Buffer,
  contentType: string,
  rateStr: string
): Promise<Buffer> {
  const filter = buildAtempoFilter(rateStr);
  if (!filter) return buffer;

  const isWav = contentType.toLowerCase().includes('wav');
  const ext = isWav ? 'wav' : 'mp3';
  const format = isWav ? 'wav' : 'mp3';
  const id = crypto.randomUUID();
  const inFile = path.join(os.tmpdir(), `kh_tts_in_${id}.${ext}`);
  const outFile = path.join(os.tmpdir(), `kh_tts_out_${id}.${ext}`);

  try {
    await fs.promises.writeFile(inFile, buffer);
    await execFileAsync('ffmpeg', [
      '-y',
      '-i',
      inFile,
      '-filter:a',
      filter,
      '-f',
      format,
      outFile,
    ]);
    const outBuf = await fs.promises.readFile(outFile);
    if (outBuf.length > 100) {
      return outBuf;
    }
    return buffer;
  } catch (err) {
    console.warn(
      'ffmpeg atempo failed, returning original audio:',
      err instanceof Error ? err.message : err
    );
    return buffer;
  } finally {
    fs.promises.unlink(inFile).catch(() => {});
    fs.promises.unlink(outFile).catch(() => {});
  }
}

/**
 * Synthesize speech using gTTS (Google Text-to-Speech) endpoint.
 */
export async function synthesizeGoogleTts(
  text: string,
  lang: string = 'km',
  rateStr: string = '+0%'
): Promise<TtsSynthesisResult> {
  const chunks = splitTextForGtts(text, 180);
  if (chunks.length === 0) {
    throw new Error('No speakable text provided for gTTS');
  }

  const buffers: Buffer[] = [];

  for (const chunk of chunks) {
    let chunkBuffer: Buffer | null = null;

    // Strategy 1: client=gtx
    try {
      const url = `https://translate.googleapis.com/translate_tts?ie=UTF-8&client=gtx&tl=${encodeURIComponent(
        lang
      )}&ttsspeed=1&q=${encodeURIComponent(chunk)}`;
      const resp = await fetch(url, {
        headers: {
          'User-Agent': CHROME_USER_AGENT,
          Referer: 'https://translate.google.com/',
        },
        signal: AbortSignal.timeout(8000),
      });
      if (resp.ok) {
        const arr = await resp.arrayBuffer();
        if (arr.byteLength > 100) {
          chunkBuffer = Buffer.from(arr);
        }
      }
    } catch (err) {
      console.warn('gTTS gtx error:', err instanceof Error ? err.message : err);
    }

    // Strategy 2: client=tw-ob
    if (!chunkBuffer) {
      try {
        const url = `https://translate.googleapis.com/translate_tts?ie=UTF-8&client=tw-ob&tl=${encodeURIComponent(
          lang
        )}&ttsspeed=1&q=${encodeURIComponent(chunk)}`;
        const resp = await fetch(url, {
          headers: {
            'User-Agent': CHROME_USER_AGENT,
          },
          signal: AbortSignal.timeout(8000),
        });
        if (resp.ok) {
          const arr = await resp.arrayBuffer();
          if (arr.byteLength > 100) {
            chunkBuffer = Buffer.from(arr);
          }
        }
      } catch (err) {
        console.warn('gTTS tw-ob error:', err instanceof Error ? err.message : err);
      }
    }

    if (!chunkBuffer) {
      throw new Error(`gTTS failed for chunk: ${chunk.slice(0, 30)}`);
    }
    buffers.push(chunkBuffer);
  }

  const combined = Buffer.concat(buffers);
  const adjusted = await applyAudioTempo(combined, 'audio/mpeg', rateStr);

  return {
    buffer: adjusted,
    contentType: 'audio/mpeg',
  };
}

/**
 * Synthesize Khmer using Meta's official facebook/mms-tts-khm model via
 * hosted Hugging Face MMS-TTS Space.
 */
export async function synthesizeMmsMetaKhmer(
  text: string,
  rateStr: string = '+0%'
): Promise<TtsSynthesisResult> {
  const cleaned = text
    .replace(/[\u200B-\u200D\uFEFF]/g, '')
    .replace(/\s*\/\s*/g, ', ')
    .trim();
  if (!cleaned) {
    throw new Error('No text provided for Meta MMS-TTS');
  }

  const spaceBase = 'https://dpc-mmstts.hf.space';
  const predictResp = await fetch(`${spaceBase}/api/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      data: [cleaned, 'Khmer (khm)'],
      fn_index: 0,
    }),
    signal: AbortSignal.timeout(12000),
  });

  if (!predictResp.ok) {
    throw new Error(`Meta MMS-TTS predict HTTP ${predictResp.status}`);
  }

  const json = (await predictResp.json()) as {
    data?: Array<{ name?: string } | string>;
  };
  const firstOut = json?.data?.[0];
  const filePath =
    typeof firstOut === 'string' ? firstOut : firstOut?.name || '';

  if (!filePath) {
    throw new Error('Meta MMS-TTS returned no audio file path');
  }

  const audioResp = await fetch(`${spaceBase}/file=${filePath}`, {
    signal: AbortSignal.timeout(10000),
  });
  if (!audioResp.ok) {
    throw new Error(`Meta MMS-TTS audio download HTTP ${audioResp.status}`);
  }

  const arr = await audioResp.arrayBuffer();
  if (arr.byteLength < 200) {
    throw new Error('Meta MMS-TTS returned empty audio buffer');
  }

  const rawBuf = Buffer.from(arr);
  const contentType = audioResp.headers.get('content-type') || 'audio/wav';
  const adjusted = await applyAudioTempo(rawBuf, contentType, rateStr);

  return {
    buffer: adjusted,
    contentType,
  };
}

/**
 * Dispatcher for extra Khmer TTS engines (MMS-TTS & gTTS) with resilient fallback.
 */
export async function synthesizeCustomKhmerTts(
  text: string,
  voice: string,
  rateStr: string = '+0%'
): Promise<TtsSynthesisResult | null> {
  if (voice === 'km-KH-gTTS') {
    return synthesizeGoogleTts(text, 'km', rateStr);
  }
  if (voice === 'km-KH-MMS-Neural') {
    try {
      return await synthesizeMmsMetaKhmer(text, rateStr);
    } catch (err) {
      console.warn(
        'Meta MMS-TTS failed, falling back to gTTS:',
        err instanceof Error ? err.message : err
      );
      return synthesizeGoogleTts(text, 'km', rateStr);
    }
  }

  return null;
}
