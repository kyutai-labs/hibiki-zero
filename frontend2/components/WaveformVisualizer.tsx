"use client";

import {
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
  useState,
} from "react";

export interface TextItem {
  text: string;
  time: number; // timestamp in ms when the text was received
}

export interface WaveformVisualizerRef {
  addAudioChunk: (chunk: Float32Array) => void;
}

interface WaveformVisualizerProps {
  analyzerNode: AnalyserNode | null;
  width?: number;
  height?: number;
  waveformColor?: string;
  textColor?: string;
  displayDuration?: number; // Duration in seconds to display
  sampleRate?: number;
  backgroundColor?: string;
  textItems?: TextItem[];
}

interface AmplitudeSnapshot {
  time: number; // timestamp in ms
  min: number;
  max: number;
}

const WaveformVisualizer = forwardRef<
  WaveformVisualizerRef,
  WaveformVisualizerProps
>(
  (
    {
      analyzerNode,
      width = 800,
      height = 200,
      waveformColor = "#22c55e",
      textColor = "#ffffff",
      displayDuration = 5,
      sampleRate = 24000,
      backgroundColor = "transparent",
      textItems = [],
    },
    ref,
  ) => {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const animationFrameRef = useRef<number>(0);
    const audioBufferRef = useRef<Float32Array>(new Float32Array(0));
    const snapshotsRef = useRef<AmplitudeSnapshot[]>([]);
    const [displaySize, setDisplaySize] = useState({ width, height });

    useImperativeHandle(ref, () => ({
      addAudioChunk: (chunk: Float32Array) => {
        const combined = new Float32Array(
          audioBufferRef.current.length + chunk.length,
        );
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

      const observer = new ResizeObserver((entries) => {
        const entry = entries[0];
        if (entry) {
          setDisplaySize({
            width: entry.contentRect.width,
            height: entry.contentRect.height || height,
          });
        }
      });

      observer.observe(canvas);
      return () => observer.disconnect();
    }, [height]);

    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      // Use the actual display size for internal resolution
      const dpr = typeof window !== "undefined" ? window.devicePixelRatio : 1;
      canvas.width = displaySize.width * dpr;
      canvas.height = displaySize.height * dpr;
      ctx.scale(dpr, dpr);

      const render = () => {
        const drawWidth = displaySize.width;
        const drawHeight = displaySize.height - 20;
        const now = Date.now();

        // Maintain constant velocity regardless of width
        const effectiveDuration = displayDuration * (drawWidth / width);

        // Clear canvas
        ctx.clearRect(0, 0, displaySize.width, displaySize.height);
        if (backgroundColor !== "transparent") {
          ctx.fillStyle = backgroundColor;
          ctx.fillRect(0, 0, drawWidth, drawHeight);
        }

        // Draw center line
        ctx.strokeStyle = "#333";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, drawHeight / 2);
        ctx.lineTo(drawWidth, drawHeight / 2);
        ctx.stroke();

        if (analyzerNode === null) {
          return;
        }

        // Draw waveform
        ctx.strokeStyle = waveformColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();

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

        // Trim snapshots
        const cutoff = now - effectiveDuration * 1000;
        snapshotsRef.current = snapshotsRef.current.filter(
          (s) => s.time >= cutoff,
        );

        // Draw accumulated snapshots
        let firstPoint = true;
        for (const snap of snapshotsRef.current) {
          const secondsAgo = (now - snap.time) / 1000;
          const x = (1 - secondsAgo / effectiveDuration) * drawWidth;
          const yMin = ((1 - snap.min) / 2) * drawHeight;
          const yMax = ((1 - snap.max) / 2) * drawHeight;

          if (firstPoint) {
            ctx.moveTo(x, yMin);
            firstPoint = false;
          } else {
            ctx.lineTo(x, yMin);
          }
          ctx.lineTo(x, yMax);
        }

        ctx.stroke();

        // Draw text items based on timestamps, pushing overlapping items right
        ctx.font = "20px sans-serif";
        ctx.fillStyle = textColor;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        const padding = 2;
        const sortedItems = [...textItems]
          .filter((item) => (now - item.time) / 1000 <= effectiveDuration * 2)
          .sort((a, b) => a.time - b.time);

        // Merge consecutive items that don't start with a space (partial words)
        const mergedItems: { text: string; time: number }[] = [];
        for (const item of sortedItems) {
          if (mergedItems.length > 0 && !item.text.startsWith(" ")) {
            mergedItems[mergedItems.length - 1].text += item.text;
          } else {
            mergedItems.push({ text: item.text, time: item.time });
          }
        }

        let minNextLeftEdge = -Infinity;

        for (const item of mergedItems) {
          const age = (now - item.time) / 1000;
          const naturalX = (1 - age / effectiveDuration) * drawWidth;
          const textWidth = ctx.measureText(item.text).width;
          const x = Math.max(naturalX, minNextLeftEdge);
          ctx.fillText(item.text, x, drawHeight);
          minNextLeftEdge = x + textWidth + padding;
        }

        animationFrameRef.current = requestAnimationFrame(render);
      };

      render();

      return () => {
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
        }
      };
    }, [
      displaySize,
      waveformColor,
      textColor,
      displayDuration,
      sampleRate,
      analyzerNode,
      backgroundColor,
      textItems,
      width,
    ]);

    return (
      <canvas
        ref={canvasRef}
        className="w-full"
        style={{ height: `${height}px`, maxWidth: `${width}px` }}
      />
    );
  },
);

WaveformVisualizer.displayName = "WaveformVisualizer";

export default WaveformVisualizer;
