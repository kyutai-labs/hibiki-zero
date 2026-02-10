"use client";

import { useEffect, useRef, forwardRef, useImperativeHandle } from "react";

export interface TextItem {
  text: string;
  time: number; // timestamp in ms when the text was received
}

export interface WaveformVisualizerRef {
  addAudioChunk: (chunk: Float32Array) => void;
}

interface WaveformVisualizerProps {
  width?: number;
  height?: number;
  waveformColor?: string;
  textColor?: string;
  displayDuration?: number; // Duration in seconds to display
  sampleRate?: number;
  analyzerNode?: AnalyserNode | null;
  backgroundColor?: string;
  textItems?: TextItem[];
}

interface AmplitudeSnapshot {
  time: number; // timestamp in ms
  min: number;
  max: number;
}

const WaveformVisualizer = forwardRef<WaveformVisualizerRef, WaveformVisualizerProps>(
  (
    {
      width = 800,
      height = 200,
      waveformColor = "#22c55e",
      textColor = "#ffffff",
      displayDuration = 5,
      sampleRate = 24000,
      analyzerNode = null,
      backgroundColor = "transparent",
      textItems = [],
    },
    ref,
  ) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number>(0);
    const audioBufferRef = useRef<Float32Array>(new Float32Array(0));
    const snapshotsRef = useRef<AmplitudeSnapshot[]>([]);

    useImperativeHandle(ref, () => ({
      addAudioChunk: (chunk: Float32Array) => {
        const combined = new Float32Array(audioBufferRef.current.length + chunk.length);
        combined.set(audioBufferRef.current);
        combined.set(chunk, audioBufferRef.current.length);
        audioBufferRef.current = combined;

        // Trim buffer to only keep displayDuration worth of audio
        const maxSamples = displayDuration * sampleRate;
        if (audioBufferRef.current.length > maxSamples) {
          audioBufferRef.current = audioBufferRef.current.slice(-maxSamples);
        }
      },
    }));

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Set canvas size
      canvas.width = width;
      canvas.height = height;

      const render = () => {
        const now = Date.now();

        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        if (backgroundColor !== "transparent") {
          ctx.fillStyle = backgroundColor;
          ctx.fillRect(0, 0, width, height);
        }

        // Draw center line
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();

        // Draw waveform
        ctx.strokeStyle = waveformColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

        if (analyzerNode) {
          // Sample the analyser and accumulate snapshots over time
          const bufferLength = analyzerNode.frequencyBinCount;
          const dataArray = new Uint8Array(bufferLength);
          analyzerNode.getByteTimeDomainData(dataArray);

          // Compute min/max amplitude for this frame
          let min = 1;
          let max = -1;
          for (let i = 0; i < bufferLength; i++) {
            const sample = (dataArray[i] - 128) / 128;
            if (sample < min) min = sample;
            if (sample > max) max = sample;
          }

          snapshotsRef.current.push({ time: now, min, max });

          // Trim to displayDuration
          const cutoff = now - displayDuration * 1000;
          snapshotsRef.current = snapshotsRef.current.filter((s) => s.time >= cutoff);

          // Draw accumulated snapshots
          let firstPoint = true;
          for (const snap of snapshotsRef.current) {
            const age = (now - snap.time) / 1000; // seconds ago
            const x = (1 - age / displayDuration) * width;
            const yMin = ((1 - snap.min) / 2) * height;
            const yMax = ((1 - snap.max) / 2) * height;

            if (firstPoint) {
              ctx.moveTo(x, yMin);
              firstPoint = false;
            } else {
              ctx.lineTo(x, yMin);
            }
            ctx.lineTo(x, yMax);
          }
        } else {
          // Min/max approach for buffered audio (many samples per pixel)
          const samplesPerPixel = (displayDuration * sampleRate) / width;
          const audioData = audioBufferRef.current;
          let firstPoint = true;
          for (let x = 0; x < width; x++) {
            const sampleIndex = Math.floor(x * samplesPerPixel);
            if (sampleIndex >= audioData.length) break;

            const endIndex = Math.min(sampleIndex + Math.ceil(samplesPerPixel), audioData.length);

            let min = 1;
            let max = -1;
            for (let i = sampleIndex; i < endIndex; i++) {
              const sample = audioData[i];
              if (sample < min) min = sample;
              if (sample > max) max = sample;
            }

            const yMin = ((1 - min) / 2) * height;
            const yMax = ((1 - max) / 2) * height;

            if (firstPoint) {
              ctx.moveTo(x, yMin);
              firstPoint = false;
            } else {
              ctx.lineTo(x, yMin);
            }
            ctx.lineTo(x, yMax);
          }
        }

        ctx.stroke();

        // Draw text items based on timestamps
        ctx.font = "14px monospace";
        ctx.fillStyle = textColor;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";

        for (const item of textItems) {
          const age = (now - item.time) / 1000;
          if (age > displayDuration * 1.1) continue;
          const x = (1 - age / displayDuration) * width;
          ctx.fillText(item.text, x, height - 20);
        }

        animationFrameRef.current = requestAnimationFrame(render);
      };

      render();

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }, [width, height, waveformColor, textColor, displayDuration, sampleRate, analyzerNode, backgroundColor, textItems]);

    return (
      <canvas
        ref={canvasRef}
        style={{ width: `${width}px`, height: `${height}px` }}
      />
    );
  }
);

WaveformVisualizer.displayName = "WaveformVisualizer";

export default WaveformVisualizer;
