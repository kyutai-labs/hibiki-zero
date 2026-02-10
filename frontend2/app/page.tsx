"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import useWebSocket, { ReadyState } from "react-use-websocket";
import { useAudioProcessor } from "./useAudioProcessor";
import { Circle } from "lucide-react";
import { clsx } from "clsx";
import WaveformVisualizer, {
  WaveformVisualizerRef,
} from "../components/WaveformVisualizer";

export default function Home() {
  const [shouldConnect, setShouldConnect] = useState(false);
  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();
  const [firstTime, setFirstTime] = useState(true);

  const [wordsReceived, setWordsReceived] = useState<
    { text: string; time: number }[]
  >([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [stepsSinceLastWord, setStepsSinceLastWord] = useState(0);

  const userVisualizerRef = useRef<WaveformVisualizerRef>(null);
  const hibikiVisualizerRef = useRef<WaveformVisualizerRef>(null);

  const webSocketUrl = "ws://localhost:8998/api/chat";

  const { sendMessage, readyState, lastMessage } = useWebSocket(
    webSocketUrl,
    {
      onError: (event) => {
        console.error("WebSocket error:", event);
        setErrors((prev) => [
          ...prev,
          `Could not connect to the translation server at ${webSocketUrl}`,
        ]);
        shutdownAudio();
        setShouldConnect(false);
      },
    },
    shouldConnect,
  );

  // const connectionStatus = {
  //   [ReadyState.CONNECTING]: "Connecting",
  //   [ReadyState.OPEN]: "Connection open",
  //   [ReadyState.CLOSING]: "Connection closing",
  //   [ReadyState.CLOSED]: "Connection closed",
  //   [ReadyState.UNINSTANTIATED]: "Connection uninstantiated",
  // }[readyState];

  const onAudioReceivedFromMic = useCallback(
    (opusAudio: Uint8Array) => {
      const message = new Uint8Array(opusAudio.length + 1);
      message[0] = 1;
      message.set(opusAudio, 1);
      sendMessage(message);
    },
    [sendMessage],
  );

  const { setupAudio, shutdownAudio, audioProcessor, processingDelaySec } =
    useAudioProcessor(onAudioReceivedFromMic);

  useEffect(() => {
    if (lastMessage === null) return;
    // We need async for decodeFromBlob
    const handleMessage = async () => {
      if (!(lastMessage.data instanceof Blob)) {
        console.error("Expected Blob data, but received:", lastMessage.data);
        return;
      }
      const lastMessageBytes = new Uint8Array(
        await lastMessage.data.arrayBuffer(),
      );
      const kind = lastMessageBytes[0];
      const dataBytes = lastMessageBytes.slice(1);

      if (kind === 2) {
        // Text data
        const textDecoder = new TextDecoder();
        const text = textDecoder.decode(dataBytes);
        const TEXT_STREAM_OFFSET_MS = 160; // Hibiki's audio is delayed by two frames compared to the text.
        setWordsReceived((prev) => [
          ...prev,
          { text, time: Date.now() + TEXT_STREAM_OFFSET_MS },
        ]);
        setStepsSinceLastWord(0);
      } else if (kind === 1) {
        // Audio data
        const ap = audioProcessor.current;
        if (!ap) return;

        ap.decoder.postMessage({
          command: "decode",
          pages: dataBytes,
        });
        setStepsSinceLastWord((prev) => prev + 1);
      }
    };
    handleMessage();
  }, [audioProcessor, lastMessage]);

  useEffect(() => {
    if (readyState === ReadyState.OPEN && shouldConnect) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFirstTime(false);
      setErrors([]);
    }
  }, [readyState, shouldConnect]);

  const onConnectButtonPress = async () => {
    // If we're not connected yet
    if (!shouldConnect) {
      const mediaStream = await askMicrophoneAccess();
      // If we have access to the microphone:
      if (mediaStream) {
        setupAudio(mediaStream);
        setShouldConnect(true);
        setWordsReceived([]);
        setStepsSinceLastWord(0);
      }
    } else {
      setShouldConnect(false);
      shutdownAudio();
      setErrors([]); // Clear previous connection errors
    }
  };

  const allErrors = errors.concat(
    microphoneAccess === "refused"
      ? [
          "Microphone access was refused. Please allow access and refresh the page.",
        ]
      : [],
    processingDelaySec > 0.5
      ? [
          `The model is ${processingDelaySec.toFixed(1)}s behind. Perhaps a network issue?`,
        ]
      : [],
  );

  return (
    <div className="flex min-h-screen justify-center bg-background text-textgray text-sm">
      <main className="flex min-h-screen w-xl max-w-screen flex-col items-center gap-4 py-10 px-4 bg-background sm:items-start">
        <h1 className="text-5xl text-green pb-1">Hibiki-Zero</h1>
        <div className="flex flex-col gap-2">
          <p>
            Kyutai&apos;s real-time speech translation model.
            {/* TODO link to blog and code */}
          </p>
          <p>
            Hibiki-Zero translates into English from French, Spanish, German,
            Portugese and Italian.
          </p>
          <p>Use headphones for best results.</p>
        </div>

        <div className="w-full flex flex-row items-center justify-center">
          <button
            className={clsx(
              "flex flex-row items-center justify-between gap-2 cursor-pointer w-40 px-2 py-2",
              "text-xl",
              "border border-dashed",
              readyState === ReadyState.OPEN
                ? "border-white text-white"
                : "border-green text-green",
            )}
            onClick={() => onConnectButtonPress()}
          >
            <span>
              {readyState === ReadyState.OPEN ? "Translating" : "Translate"}
            </span>
            {readyState === ReadyState.OPEN && (
              <Circle
                size={24}
                color="var(--red)"
                fill="var(--red)"
                className="animate-pulse-recording"
              />
            )}
            {!(readyState === ReadyState.OPEN) && (
              <Circle
                onClick={() => onConnectButtonPress()}
                size={24}
                color="var(--green)"
              />
            )}
          </button>
        </div>
        {allErrors.length > 0 && (
          <div>
            {allErrors.map((error, i) => (
              <p className="text-red" key={i}>
                {error}
              </p>
            ))}
          </div>
        )}
        {!firstTime && (
          <div className="w-full flex flex-col gap-4">
            <div className="relative flex flex-col">
              <span className="absolute top-2 left-2 text-textgray text-xs uppercase tracking-wider z-10 font-medium">
                You
              </span>
              <WaveformVisualizer
                ref={userVisualizerRef}
                width={800}
                height={120}
                waveformColor="#ffffff"
                textColor="#ffffff"
                displayDuration={4}
                analyzerNode={audioProcessor.current?.inputAnalyser || null}
                backgroundColor="transparent"
              />
            </div>
            <div className="relative flex flex-col">
              <span className="absolute top-2 left-2 text-textgray text-xs uppercase tracking-wider z-10 font-medium">
                Hibiki-Zero
              </span>
              <WaveformVisualizer
                ref={hibikiVisualizerRef}
                width={800}
                height={120}
                waveformColor="#39F2AE"
                textColor="#39F2AE"
                displayDuration={4}
                analyzerNode={audioProcessor.current?.outputAnalyser || null}
                backgroundColor="transparent"
                textItems={wordsReceived}
              />
            </div>
          </div>
        )}
        {!firstTime && (
          <div className="bg-gray my-4 p-4 min-h-40 w-full">
            {readyState === ReadyState.OPEN && wordsReceived.length === 0 ? (
              <span className="text-textgray">
                Speak to see your words translated...
              </span>
            ) : (
              <>
                <span>{wordsReceived.map((w) => w.text).join("")}</span>
              </>
            )}
            <span>
              {" "}
              {Array.from({
                length: Math.floor(stepsSinceLastWord / 25),
              }).map((_, i) => (
                <span className="text-textgray" key={i}>
                  &middot;{" "}
                </span>
              ))}
            </span>
          </div>
        )}
      </main>
    </div>
  );
}
