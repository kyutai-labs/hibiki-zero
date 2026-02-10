"use client";

import { useCallback, useEffect, useState } from "react";
import { useMicrophoneAccess } from "./useMicrophoneAccess";
import useWebSocket from "react-use-websocket";
import { useAudioProcessor } from "./useAudioProcessor";
import { Circle } from "lucide-react";

export default function Home() {
  const [shouldConnect, setShouldConnect] = useState(false);
  const { microphoneAccess, askMicrophoneAccess } = useMicrophoneAccess();
  const [firstTime, setFirstTime] = useState(true);

  const [wordsReceived, setWordsReceived] = useState<string[]>([]);
  const [pausePrediction, setPausePrediction] = useState(0.0);
  const [stepsSinceLastWord, setStepsSinceLastWord] = useState(0);

  const webSocketUrl = "ws://localhost:8998/api/chat";

  const { sendMessage, readyState, lastMessage } = useWebSocket(
    webSocketUrl,
    {},
    shouldConnect,
  );

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
        setWordsReceived((prev) => [...prev, text]);
        setStepsSinceLastWord(0);
      } else if (kind === 1) {
        // Audio data
        const ap = audioProcessor.current;
        if (!ap) return;

        ap.decoder.postMessage({
          command: "decode",
          pages: dataBytes,
        });
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
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center gap-2 py-32 px-16 bg-background sm:items-start">
        <h1 className="font-bold text-5xl">Hibiki-Zero</h1>
        <div
          className="flex flex-row items-center justify-end gap-2 cursor-pointer"
          onClick={() => onConnectButtonPress()}
        >
          <span>{shouldConnect ? "Transcribing" : "Stopped"}</span>
          {shouldConnect && (
            <Circle
              size={30}
              color="var(--red)"
              fill="var(--red)"
              className="animate-pulse-recording"
            />
          )}
          {!shouldConnect && (
            <Circle
              onClick={() => onConnectButtonPress()}
              size={30}
              color="white"
            />
          )}
        </div>
        <div>
          <p>{wordsReceived.join("")}</p>
        </div>
      </main>
    </div>
  );
}
