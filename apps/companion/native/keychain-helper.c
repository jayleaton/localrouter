// localrouter-keychain: the one process that reads and writes LocalRouter's secrets in the macOS keychain.
//
// Why a helper: a keychain item made by an app without an Apple Team ID is partitioned to the code hash (cdhash) of
// the process that made it, so each new LocalRouter build (a new cdhash, even signed by the same certificate) gets a
// "wants to use your confidential information" prompt, and "Always Allow" does not survive the next update. This
// helper is installed once outside the app bundle (~/Library/Application Support/LocalRouter/bin) and is not replaced
// by app updates, so the process touching the keychain keeps its cdhash and the item's partition keeps matching.
//
// It answers only a parent process that satisfies LocalRouter's designated requirement (LR_APP_REQUIREMENT, the
// bundle id and the signing certificate, compiled in): run from a shell, or by any other program, it refuses.
//
//   localrouter-keychain version
//   localrouter-keychain get SERVICE ACCOUNT        the secret on stdout; exit 4 when there is none
//   localrouter-keychain set SERVICE ACCOUNT        the secret on stdin (at most 64 KiB)
//   localrouter-keychain delete SERVICE ACCOUNT
//
// Exit codes: 0 ok, 1 keychain error (message on stderr), 2 usage, 3 refused (not LocalRouter), 4 not found,
// 5 the keychain would have to ask (LR_KEYCHAIN_NO_UI=1 only: tests use it to prove no prompt happens).
#include <CoreFoundation/CoreFoundation.h>
#include <Security/Security.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "requirement.h" // #define LR_APP_REQUIREMENT "identifier \"…\" and certificate leaf = H\"…\"" (package-mac.sh)

// Bumped only when the helper must be replaced on installed Macs (that costs each Mac one keychain prompt).
#define HELPER_VERSION "1"
#define MAX_SECRET (64 * 1024)

#ifdef LR_NOISE // CI only: a "previous build" whose helper differs in bytes, to prove the installed one is kept
__attribute__((used)) static const char lr_noise[] = LR_NOISE;
#endif

static int parent_is_localrouter(void) {
  pid_t parent = getppid();
  CFNumberRef pid = CFNumberCreate(NULL, kCFNumberIntType, &parent);
  const void *keys[] = {kSecGuestAttributePid};
  const void *values[] = {pid};
  CFDictionaryRef attributes = CFDictionaryCreate(NULL, keys, values, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  SecCodeRef code = NULL;
  SecRequirementRef requirement = NULL;
  int ok = SecCodeCopyGuestWithAttributes(NULL, attributes, kSecCSDefaultFlags, &code) == errSecSuccess &&
           SecRequirementCreateWithString(CFSTR(LR_APP_REQUIREMENT), kSecCSDefaultFlags, &requirement) == errSecSuccess &&
           SecCodeCheckValidity(code, kSecCSDefaultFlags, requirement) == errSecSuccess;
  if (requirement) CFRelease(requirement);
  if (code) CFRelease(code);
  CFRelease(attributes);
  CFRelease(pid);
  return ok;
}

static CFMutableDictionaryRef query(const char *service, const char *account) {
  CFMutableDictionaryRef q = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  CFStringRef s = CFStringCreateWithCString(NULL, service, kCFStringEncodingUTF8);
  CFStringRef a = CFStringCreateWithCString(NULL, account, kCFStringEncodingUTF8);
  CFDictionarySetValue(q, kSecClass, kSecClassGenericPassword);
  CFDictionarySetValue(q, kSecAttrService, s);
  CFDictionarySetValue(q, kSecAttrAccount, a);
  CFRelease(s);
  CFRelease(a);
  return q;
}

static int fail(OSStatus status, const char *what) {
  if (status == errSecItemNotFound) return 4;
  if (status == errSecInteractionNotAllowed || status == errSecInteractionRequired) {
    fprintf(stderr, "%s: the keychain would have to ask (%d)\n", what, (int)status);
    return 5;
  }
  CFStringRef message = SecCopyErrorMessageString(status, NULL);
  char text[256] = "";
  if (message) {
    CFStringGetCString(message, text, sizeof text, kCFStringEncodingUTF8);
    CFRelease(message);
  }
  fprintf(stderr, "%s: %s (%d)\n", what, text, (int)status);
  return 1;
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "version") == 0) {
    printf("%s\n", HELPER_VERSION);
    return 0;
  }
  if (argc != 4 || (strcmp(argv[1], "get") && strcmp(argv[1], "set") && strcmp(argv[1], "delete"))) {
    fprintf(stderr, "usage: localrouter-keychain version | get|set|delete SERVICE ACCOUNT\n");
    return 2;
  }
  if (!parent_is_localrouter()) {
    fprintf(stderr, "localrouter-keychain answers only LocalRouter\n");
    return 3;
  }
  if (getenv("LR_KEYCHAIN_NO_UI") && strcmp(getenv("LR_KEYCHAIN_NO_UI"), "1") == 0) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    SecKeychainSetUserInteractionAllowed(false);
#pragma clang diagnostic pop
  }
  CFMutableDictionaryRef q = query(argv[2], argv[3]);
  int code = 0;
  if (strcmp(argv[1], "get") == 0) {
    CFDictionarySetValue(q, kSecReturnData, kCFBooleanTrue);
    CFDictionarySetValue(q, kSecMatchLimit, kSecMatchLimitOne);
    CFTypeRef data = NULL;
    OSStatus status = SecItemCopyMatching(q, &data);
    if (status == errSecSuccess && data) {
      fwrite(CFDataGetBytePtr(data), 1, (size_t)CFDataGetLength(data), stdout);
      CFRelease(data);
    } else code = fail(status, "get");
  } else if (strcmp(argv[1], "set") == 0) {
    static unsigned char buffer[MAX_SECRET + 1];
    size_t n = fread(buffer, 1, sizeof buffer, stdin);
    if (n > MAX_SECRET) {
      fprintf(stderr, "set: the secret is longer than %d bytes\n", MAX_SECRET);
      code = 2;
    } else {
      CFDataRef value = CFDataCreate(NULL, buffer, (CFIndex)n);
      CFMutableDictionaryRef change = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
      CFDictionarySetValue(change, kSecValueData, value);
      OSStatus status = SecItemUpdate(q, change);
      if (status == errSecItemNotFound) {
        CFDictionarySetValue(q, kSecValueData, value);
        CFDictionarySetValue(q, kSecAttrLabel, CFSTR("LocalRouter"));
        status = SecItemAdd(q, NULL);
      }
      if (status != errSecSuccess) code = fail(status, "set");
      CFRelease(change);
      CFRelease(value);
    }
  } else {
    OSStatus status = SecItemDelete(q);
    if (status != errSecSuccess && status != errSecItemNotFound) code = fail(status, "delete");
  }
  CFRelease(q);
  return code;
}
