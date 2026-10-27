// localrouter-presence: Touch ID for LocalRouter, in its own process so the prompt can be dismissed.
//
// Why a helper: Electron's systemPreferences.promptTouchID cannot be cancelled, so a stop (⌘⇧⎋) or a Deny elsewhere
// left the Touch ID sheet on screen, and canPromptTouchID says only "no", never why. This asks LocalAuthentication
// directly: `check` says whether Touch ID can be used now and, when not, LAError's code (no finger enrolled, the lid
// closed, locked out); `verify` shows the prompt and [LAContext invalidate]s it on SIGTERM or when stdin closes.
//
//   localrouter-presence check     {"biometrics":true,"code":0} or {"biometrics":false,"code":-7}
//   localrouter-presence verify    the reason on stdin's first line; stdin then stays open until the answer
//
// verify's exit codes: 0 verified, 1 not (cancelled, failed, refused), 2 usage, 3 dismissed (SIGTERM or stdin closed),
// 4 Touch ID cannot be used now (its code on stdout).
//
// It also answers whether macOS lets LocalRouter send Apple Events to an app (Automation), for the same reason it
// lives here: a helper LocalRouter starts counts as LocalRouter for macOS's privacy checks. During a task nothing may
// open "LocalRouter wants to control …", so every Apple Event is preceded by `automation`, which never asks; only
// the app's setup runs `automation-ask`, the one moment macOS may show its prompt.
//
//   localrouter-presence automation BUNDLE_ID       {"status":0} granted, -1743 denied, -1744 would ask, -600 not running
//   localrouter-presence automation-ask BUNDLE_ID   the same, asking the person when macOS has no answer yet
//
// System Events is a background app that is often not running: it is started hidden first, so the answer is never
// "not running" for it.
#import <AppKit/AppKit.h>
#import <CoreServices/CoreServices.h>
#import <Foundation/Foundation.h>
#import <LocalAuthentication/LocalAuthentication.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

static LAContext *context;
static volatile int dismissed = 0;

static void dismiss(void) {
  if (dismissed) return;
  dismissed = 1;
  [context invalidate];
  // The reply comes with LAErrorAppCancel; if it does not, leave anyway.
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 2 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ exit(3); });
}

static dispatch_source_t on_signal(int sig) {
  signal(sig, SIG_IGN);
  dispatch_source_t s = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, sig, 0, dispatch_get_main_queue());
  dispatch_source_set_event_handler(s, ^{ dismiss(); });
  dispatch_resume(s);
  return s;
}

// Starts an app by bundle id without bringing it forward or showing it, and waits (up to 5 s) until it runs.
static void launch_hidden(NSString *bundle) {
  if ([NSRunningApplication runningApplicationsWithBundleIdentifier:bundle].count) return;
  NSURL *url = [[NSWorkspace sharedWorkspace] URLForApplicationWithBundleIdentifier:bundle];
  if (!url) return;
  NSWorkspaceOpenConfiguration *config = [NSWorkspaceOpenConfiguration configuration];
  config.activates = NO;
  config.hides = YES;
  config.addsToRecentItems = NO;
  dispatch_semaphore_t done = dispatch_semaphore_create(0);
  [[NSWorkspace sharedWorkspace] openApplicationAtURL:url configuration:config completionHandler:^(NSRunningApplication *app, NSError *error) {
    (void)app;
    (void)error;
    dispatch_semaphore_signal(done);
  }];
  dispatch_semaphore_wait(done, dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_SEC));
}

// AEDeterminePermissionToAutomateTarget for the app with that bundle id. The check uses wildcards (any event); the ask
// names the event LocalRouter sends most (core/getd, a property read), since macOS asks only for a concrete event.
// The consent is per target app, so either covers every event LocalRouter sends it.
static int automation_status(const char *bundle, Boolean ask) {
  NSString *bid = [NSString stringWithUTF8String:bundle];
  if ([bid isEqualToString:@"com.apple.systemevents"]) launch_hidden(bid);
  AEAddressDesc target;
  OSErr made = AECreateDesc(typeApplicationBundleID, bundle, (Size)strlen(bundle), &target);
  if (made != noErr) {
    printf("{\"status\":%d}\n", (int)made);
    return 0;
  }
  OSStatus status = noErr;
  // A just-started app takes a moment to be reachable: "not running" is retried for up to 2 s.
  for (int tries = 0; tries < 20; tries++) {
    status = ask ? AEDeterminePermissionToAutomateTarget(&target, kCoreEventClass, kAEGetData, true)
                 : AEDeterminePermissionToAutomateTarget(&target, typeWildCard, typeWildCard, false);
    if (status != procNotFound || ![bid isEqualToString:@"com.apple.systemevents"]) break;
    usleep(100 * 1000);
  }
  AEDisposeDesc(&target);
  printf("{\"status\":%d}\n", (int)status);
  return 0;
}

int main(int argc, const char **argv) {
  @autoreleasepool {
    if (argc == 3 && (strcmp(argv[1], "automation") == 0 || strcmp(argv[1], "automation-ask") == 0)) {
      return automation_status(argv[2], strcmp(argv[1], "automation-ask") == 0);
    }
    if (argc != 2) {
      fprintf(stderr, "usage: localrouter-presence check|verify|automation BUNDLE_ID|automation-ask BUNDLE_ID\n");
      return 2;
    }
    context = [[LAContext alloc] init];
    NSError *error = nil;
    BOOL can = [context canEvaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics error:&error];
    long code = can ? 0 : (long)(error ? error.code : -6);
    if (strcmp(argv[1], "check") == 0) {
      printf("{\"biometrics\":%s,\"code\":%ld}\n", can ? "true" : "false", code);
      return 0;
    }
    if (strcmp(argv[1], "verify") != 0) return 2;
    char line[1024];
    if (!fgets(line, sizeof line, stdin)) return 2;
    line[strcspn(line, "\r\n")] = 0;
    if (!can) {
      printf("{\"result\":\"unavailable\",\"code\":%ld}\n", code);
      return 4;
    }
    // Touch ID only, as the card said: no "Use Password…" button.
    context.localizedFallbackTitle = @"";
    dispatch_source_t term = on_signal(SIGTERM), intr = on_signal(SIGINT);
    (void)term;
    (void)intr;
    // The app closing stdin (it quit, or the approval ended) dismisses the prompt too.
    dispatch_source_t input = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, STDIN_FILENO, 0, dispatch_get_main_queue());
    dispatch_source_set_event_handler(input, ^{
      char buf[256];
      if (read(STDIN_FILENO, buf, sizeof buf) <= 0) {
        dispatch_source_cancel(input);
        dismiss();
      }
    });
    dispatch_resume(input);
    NSString *reason = [NSString stringWithUTF8String:line];
    if (reason.length == 0) reason = @"approve this action";
    [context evaluatePolicy:LAPolicyDeviceOwnerAuthenticationWithBiometrics localizedReason:reason reply:^(BOOL ok, NSError *e) {
      dispatch_async(dispatch_get_main_queue(), ^{
        if (dismissed) {
          printf("{\"result\":\"dismissed\"}\n");
          fflush(stdout);
          exit(3);
        }
        printf("{\"result\":\"%s\",\"code\":%ld}\n", ok ? "verified" : "not", ok ? 0L : (long)e.code);
        fflush(stdout);
        exit(ok ? 0 : 1);
      });
    }];
    // A person who never answers: Touch ID gives up on its own, and this does at the approval's limit (2 min).
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 130 * NSEC_PER_SEC), dispatch_get_main_queue(), ^{ dismiss(); });
    dispatch_main();
  }
}
