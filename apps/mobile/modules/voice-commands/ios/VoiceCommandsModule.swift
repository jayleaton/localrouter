import AVFoundation
import ExpoModulesCore
import Speech

/**
 On-device speech recognition over microphone audio the app already
 captures (passed in with `append`), for spoken app commands. It never opens
 the microphone or touches the shared audio engine, and it refuses to run
 without on-device recognition, so muted audio never leaves the phone.
 JavaScript decides what counts as a command, and when there is nothing to
 hear (`idle`); a task runs only while audio is being appended.
 */
public class VoiceCommandsModule: Module {
  /** Recognition tasks end on their own after long speech or silence; each window is renewed. */
  private static let windowSeconds: Double = 50
  /**
   iOS can refuse a task started right after one ends or fails, so a failed
   window is renewed after a pause. A recognizer that keeps failing (its
   model missing or being updated: "Failed to initialize recognizer") is
   retried less and less often: restarting it twice a second kept the phone
   busy and warm.
   */
  private static let retryDelaySeconds: Double = 0.5
  private static let maxRetryDelaySeconds: Double = 30

  private let queue = DispatchQueue(label: "dev.remotevoiceagent.voice-commands")
  private var recognizer: SFSpeechRecognizer?
  private var request: SFSpeechAudioBufferRecognitionRequest?
  private var task: SFSpeechRecognitionTask?
  private var format: AVAudioFormat?
  private var phrases: [String] = []
  /** Identifies the current window, so a replaced task's late results are ignored. */
  private var generation = 0
  private var running = false
  /** Windows that failed in a row, since the last result. */
  private var failures = 0
  /**
   Nothing to recognize (standby's energy gate closed, or no audio yet): no
   task runs and no window is renewed until the next `append`. A recognition
   task working on silence is the largest standby battery drain.
   */
  private var idle = true

  public func definition() -> ModuleDefinition {
    Name("VoiceCommands")
    Events("onTranscript", "onError")

    AsyncFunction("requestPermission") { (promise: Promise) in
      SFSpeechRecognizer.requestAuthorization { status in
        promise.resolve(Self.describe(status))
      }
    }

    AsyncFunction("start") { (sampleRate: Double, phrases: [String], promise: Promise) in
      self.queue.async {
        guard SFSpeechRecognizer.authorizationStatus() == .authorized,
              let recognizer = self.recognizer ?? SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
              recognizer.supportsOnDeviceRecognition,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)
        else {
          promise.resolve(false)
          return
        }
        self.recognizer = recognizer
        self.format = format
        self.phrases = phrases
        self.running = true
        // The first task starts with the first audio.
        self.idle = true
        promise.resolve(true)
      }
    }

    Function("append") { (pcm16Base64: String) in
      guard let data = Data(base64Encoded: pcm16Base64) else { return }
      self.queue.async { self.append(data) }
    }

    Function("reset") {
      self.queue.async { if self.running && !self.idle { self.beginWindow() } }
    }

    Function("idle") {
      self.queue.async { self.goIdle() }
    }

    Function("stop") {
      self.queue.async { self.stopListening() }
    }

    OnDestroy {
      self.queue.async { self.stopListening() }
    }
  }

  private func beginWindow() {
    task?.cancel()
    request?.endAudio()
    guard running, let recognizer else { return }
    idle = false
    generation += 1
    let current = generation
    let request = SFSpeechAudioBufferRecognitionRequest()
    request.shouldReportPartialResults = true
    request.requiresOnDeviceRecognition = true
    request.contextualStrings = phrases
    request.addsPunctuation = false
    request.taskHint = .confirmation
    self.request = request
    task = recognizer.recognitionTask(with: request) { [weak self] result, error in
      guard let self else { return }
      self.queue.async {
        guard current == self.generation, self.running else { return }
        if let result {
          self.failures = 0
          self.sendEvent("onTranscript", ["text": result.bestTranscription.formattedString, "isFinal": result.isFinal])
        }
        // An idle task finishing its last audio: its words still count, but nothing is renewed.
        if self.idle { return }
        if let error {
          let nsError = error as NSError
          self.sendEvent("onError", ["message": nsError.localizedDescription, "domain": nsError.domain, "code": nsError.code])
          self.generation += 1
          let retry = self.generation
          let delay = min(Self.retryDelaySeconds * pow(2, Double(min(self.failures, 6))), Self.maxRetryDelaySeconds)
          self.failures += 1
          self.queue.asyncAfter(deadline: .now() + delay) { [weak self] in
            guard let self, retry == self.generation, !self.idle else { return }
            self.beginWindow()
          }
        } else if result?.isFinal == true {
          self.beginWindow()
        }
      }
    }
    queue.asyncAfter(deadline: .now() + Self.windowSeconds) { [weak self] in
      guard let self, current == self.generation, !self.idle else { return }
      self.beginWindow()
    }
  }

  /** Ends the current task once it has recognized the audio it has, and renews nothing until more arrives. */
  private func goIdle() {
    guard running, !idle else { return }
    idle = true
    request?.endAudio()
  }

  private func append(_ data: Data) {
    guard running else { return }
    if idle { beginWindow() }
    guard let request, let format else { return }
    let frames = data.count / 2
    guard frames > 0, let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)),
          let channel = buffer.floatChannelData?[0]
    else { return }
    buffer.frameLength = AVAudioFrameCount(frames)
    data.withUnsafeBytes { raw in
      let samples = raw.bindMemory(to: Int16.self)
      for index in 0..<frames { channel[index] = Float(Int16(littleEndian: samples[index])) / 32768 }
    }
    request.append(buffer)
  }

  private func stopListening() {
    running = false
    generation += 1
    task?.cancel()
    request?.endAudio()
    task = nil
    request = nil
    idle = true
  }

  private static func describe(_ status: SFSpeechRecognizerAuthorizationStatus) -> String {
    switch status {
    case .authorized: return "granted"
    case .denied: return "denied"
    case .restricted: return "restricted"
    default: return "notDetermined"
    }
  }
}
