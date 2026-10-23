import AVFoundation
import CallKit
import ExpoModulesCore

/**
 Registers the voice session as an outgoing CallKit call, so it appears in
 CarPlay and answers car buttons. CallKit owns audio-session activation:
 JavaScript starts audio only after `onAudioActivated`. Outgoing calls need no
 PushKit.

 It also reports whether the phone is in a CarPlay car (JavaScript makes work
 mode a system call only then: on the phone, iOS's own call screen would cover
 the lock screen card) and the AirPods/headset mute gesture for a call that is
 not a system call.
 */
public class CallSessionModule: Module {
  private let controller = CXCallController()
  private lazy var provider: CXProvider = {
    let configuration = CXProviderConfiguration()
    configuration.supportsVideo = false
    configuration.maximumCallGroups = 1
    configuration.maximumCallsPerCallGroup = 1
    configuration.supportedHandleTypes = [.generic]
    configuration.includesCallsInRecents = false
    let provider = CXProvider(configuration: configuration)
    provider.setDelegate(delegate, queue: nil)
    return provider
  }()
  private lazy var delegate = CallDelegate(module: self)
  fileprivate var callId: UUID?
  /** Set while the app itself ends the call, so no end request is echoed back. */
  fileprivate var endingFromApp = false
  private var carConnected = false
  private var observers: [NSObjectProtocol] = []

  public func definition() -> ModuleDefinition {
    Name("CallSession")
    Events("onAudioActivated", "onAudioDeactivated", "onMuteChanged", "onEndRequested", "onCarChanged", "onHeadsetMuteGesture")

    OnCreate {
      self.carConnected = Self.inCar()
      let center = NotificationCenter.default
      self.observers.append(center.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { _ in
        let connected = Self.inCar()
        guard connected != self.carConnected else { return }
        self.carConnected = connected
        self.sendEvent("onCarChanged", ["connected": connected])
      })
      if #available(iOS 17.0, *) {
        // The press-to-mute gesture (AirPods stem, headset). During a system call CallKit handles it
        // as CXSetMutedCallAction instead. Otherwise the app mutes with its own gate and keeps the
        // input live, so the on-device listener can still hear the unmute phrase: the gesture is
        // reported as a toggle and the system's input mute is cleared again.
        self.observers.append(center.addObserver(forName: AVAudioApplication.inputMuteStateChangeNotification, object: nil, queue: .main) { note in
          guard self.callId == nil, (note.userInfo?[AVAudioApplication.muteStateKey] as? Bool) == true else { return }
          try? AVAudioApplication.shared.setInputMuted(false)
          self.sendEvent("onHeadsetMuteGesture", [:])
        })
      }
    }

    OnDestroy {
      self.observers.forEach { NotificationCenter.default.removeObserver($0) }
      self.observers = []
    }

    Function("isCarConnected") { Self.inCar() }

    AsyncFunction("startCall") { (displayName: String, promise: Promise) in
      if let existing = self.callId {
        promise.resolve(existing.uuidString)
        return
      }
      _ = self.provider
      let id = UUID()
      self.callId = id
      let action = CXStartCallAction(call: id, handle: CXHandle(type: .generic, value: displayName))
      action.isVideo = false
      self.controller.request(CXTransaction(action: action)) { error in
        if let error {
          self.callId = nil
          promise.reject("ERR_CALL_START", error.localizedDescription)
        } else {
          promise.resolve(id.uuidString)
        }
      }
    }

    AsyncFunction("reportConnected") {
      guard let id = self.callId else { return }
      self.provider.reportOutgoingCall(with: id, connectedAt: Date())
    }

    AsyncFunction("setMuted") { (muted: Bool, promise: Promise) in
      guard let id = self.callId else {
        promise.resolve(nil)
        return
      }
      self.controller.request(CXTransaction(action: CXSetMutedCallAction(call: id, muted: muted))) { _ in promise.resolve(nil) }
    }

    AsyncFunction("endCall") { (promise: Promise) in
      guard let id = self.callId else {
        promise.resolve(nil)
        return
      }
      self.endingFromApp = true
      self.controller.request(CXTransaction(action: CXEndCallAction(call: id))) { error in
        if error != nil {
          // The system no longer knows the call; report it ended so the UI clears.
          self.provider.reportCall(with: id, endedAt: Date(), reason: .remoteEnded)
          self.callId = nil
        }
        self.endingFromApp = false
        promise.resolve(nil)
      }
    }

    Function("isActive") { self.callId != nil }
  }

  /** CarPlay's audio port, as the route or as an available input (it is listed even while the session is inactive). */
  private static func inCar() -> Bool {
    let session = AVAudioSession.sharedInstance()
    return session.currentRoute.outputs.contains { $0.portType == .carAudio }
      || (session.availableInputs ?? []).contains { $0.portType == .carAudio }
  }

  fileprivate func emit(_ name: String, _ body: [String: Any] = [:]) {
    sendEvent(name, body)
  }

  fileprivate func startConnecting(_ id: UUID) {
    provider.reportOutgoingCall(with: id, startedConnectingAt: Date())
  }
}

private final class CallDelegate: NSObject, CXProviderDelegate {
  weak var module: CallSessionModule?

  init(module: CallSessionModule) {
    self.module = module
  }

  func providerDidReset(_ provider: CXProvider) {
    module?.callId = nil
    module?.emit("onEndRequested")
  }

  func provider(_ provider: CXProvider, perform action: CXStartCallAction) {
    // Configure before CallKit activates the session; the app never activates it itself.
    let session = AVAudioSession.sharedInstance()
    try? session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker, .allowBluetoothHFP])
    module?.startConnecting(action.callUUID)
    action.fulfill()
  }

  func provider(_ provider: CXProvider, perform action: CXEndCallAction) {
    let fromApp = module?.endingFromApp ?? false
    module?.callId = nil
    action.fulfill()
    if !fromApp { module?.emit("onEndRequested") }
  }

  func provider(_ provider: CXProvider, perform action: CXSetMutedCallAction) {
    module?.emit("onMuteChanged", ["muted": action.isMuted])
    action.fulfill()
  }

  func provider(_ provider: CXProvider, didActivate audioSession: AVAudioSession) {
    module?.emit("onAudioActivated")
  }

  func provider(_ provider: CXProvider, didDeactivate audioSession: AVAudioSession) {
    module?.emit("onAudioDeactivated")
  }
}
