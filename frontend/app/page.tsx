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

  const { setupAudio, shutdownAudio, audioProcessor } = useAudioProcessor(
    onAudioReceivedFromMic,
  );

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
        setWordsReceived((prev) => [...prev, { text, time: Date.now() }]);
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
        setFirstTime(false);
      }
    } else {
      setShouldConnect(false);
      shutdownAudio();
      setErrors([]); // Clear previous connection errors
    }
  };

  const errorsIncludingMic = errors.concat(
    microphoneAccess === "refused"
      ? [
          "Microphone access was refused. Please allow access and refresh the page.",
        ]
      : [],
  );

  return (
    <div className="flex min-h-screen items-center justify-center bg-background text-textgray text-sm">
      <main className="flex min-h-screen w-xl max-w-screen flex-col items-center gap-2 py-32 px-4 bg-background sm:items-start">
        <h1 className="text-5xl text-green pb-1">Hibiki-Zero</h1>
        <div className="flex flex-col gap-2">
          <p>
            Kyutai&apos;s real-time speech translation model. TODO link to blog
            post and code.
          </p>
          <p>
            Hibiki-Zero translates into English from French, Spanish, German,
            Portugese and Italian.
          </p>
          <p>Use headphones for best results.</p>
        </div>

        <button
          className={clsx(
            "flex flex-row items-center justify-between gap-2 cursor-pointer w-40 px-2 py-2",
            "text-xl",
            "border border-dashed",
            shouldConnect
              ? "border-white text-white"
              : "border-green text-green",
          )}
          onClick={() => onConnectButtonPress()}
        >
          <span>{shouldConnect ? "Translating" : "Translate"}</span>
          {shouldConnect && (
            <Circle
              size={24}
              color="var(--red)"
              fill="var(--red)"
              className="animate-pulse-recording"
            />
          )}
          {!shouldConnect && (
            <Circle
              onClick={() => onConnectButtonPress()}
              size={24}
              color="var(--green)"
            />
          )}
        </button>
        {errorsIncludingMic.length > 0 && (
          <div>
            {errorsIncludingMic.map((error, i) => (
              <p className="text-red" key={i}>
                {error}
              </p>
            ))}
          </div>
        )}
        {shouldConnect && audioProcessor.current && (
          <div className="w-full flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <span className="text-textgray text-xs">You</span>
              <WaveformVisualizer
                ref={userVisualizerRef}
                width={800}
                height={120}
                waveformColor="#ffffff"
                textColor="#ffffff"
                displayDuration={5}
                analyzerNode={audioProcessor.current.inputAnalyser}
                backgroundColor="transparent"
              />
            </div>
            <div className="flex flex-col gap-1">
              <span className="text-textgray text-xs">Hibiki</span>
              <WaveformVisualizer
                ref={hibikiVisualizerRef}
                width={800}
                height={120}
                waveformColor="#39F2AE"
                textColor="#39F2AE"
                displayDuration={5}
                analyzerNode={audioProcessor.current.outputAnalyser}
                backgroundColor="transparent"
                textItems={wordsReceived}
              />
            </div>
          </div>
        )}
        {!firstTime && shouldConnect && readyState === ReadyState.OPEN && (
          <div className="bg-gray my-4 p-4 min-h-40 w-full">
            {shouldConnect && (
              <>
                {wordsReceived.length === 0 ? (
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
              </>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
