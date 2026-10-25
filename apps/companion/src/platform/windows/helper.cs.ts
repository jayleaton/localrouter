/**
 * The C# half of the Windows worker, compiled once per worker start by Windows PowerShell 5.1's `Add-Type`, which uses
 * the .NET Framework's own csc: C# 5 only (no `$"…"`, `?.`, `nameof`, expression-bodied members, `out var`, tuples or
 * exception filters). It holds everything that needs UI Automation or Win32: reading the front window, acting through
 * UIA patterns, SendInput, screenshots, displays, the pointer, windows, Core Audio volume, dark mode and locking.
 *
 * Requests arrive as JSON text and are parsed here (a small reader below), so no request data is ever spliced into
 * script or code. Replies are JSON with every non-ASCII character escaped, so the pipe carries ASCII only and neither
 * side's console code page can garble a label or typed text.
 *
 * Failures that a person can act on are thrown as `LR_CODE|detail|`; the TypeScript side turns them into sentences.
 *
 * Coordinates: Lr.Hello() makes the process per-monitor DPI aware (v2) before any UIA or GDI call. UIA's
 * BoundingRectangle, GetCursorPos, SendInput's absolute coordinates, monitor rectangles and CopyFromScreen then all use
 * physical pixels of the virtual desktop, which is the one global space of `Point` on Windows. Without it Windows
 * would scale some of them per monitor and a click on a 150 % display would land beside its control.
 */
export const HELPER_CS = String.raw`
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Automation;
using Microsoft.Win32;

namespace LocalRouterWin
{
    public sealed class LrError : Exception
    {
        public LrError(string code, string detail) : base("LR_" + code + "|" + (detail ?? "").Replace("|", "/") + "|") { }
    }

    public sealed class Request
    {
        public long Id;
        public string Op;
        public Dictionary<string, object> Args;
    }

    /** A small JSON reader: objects, arrays, strings, numbers (as double), true, false, null. */
    internal sealed class JsonReader
    {
        private readonly string s;
        private int i;

        public JsonReader(string text) { s = text; i = 0; }

        public object Document()
        {
            object value = Value();
            Space();
            if (i != s.Length) throw new FormatException("trailing text in JSON");
            return value;
        }

        private void Space() { while (i < s.Length && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n')) i++; }

        private object Value()
        {
            Space();
            if (i >= s.Length) throw new FormatException("JSON ended early");
            char c = s[i];
            if (c == '{') return ReadObject();
            if (c == '[') return ReadArray();
            if (c == '"') return ReadString();
            if (Word("true")) return true;
            if (Word("false")) return false;
            if (Word("null")) return null;
            return ReadNumber();
        }

        private bool Word(string word)
        {
            if (string.CompareOrdinal(s, i, word, 0, word.Length) != 0) return false;
            i += word.Length;
            return true;
        }

        private Dictionary<string, object> ReadObject()
        {
            Dictionary<string, object> map = new Dictionary<string, object>();
            i++;
            Space();
            if (i < s.Length && s[i] == '}') { i++; return map; }
            while (true)
            {
                Space();
                if (i >= s.Length || s[i] != '"') throw new FormatException("expected a key in JSON");
                string key = ReadString();
                Space();
                if (i >= s.Length || s[i] != ':') throw new FormatException("expected : in JSON");
                i++;
                map[key] = Value();
                Space();
                if (i < s.Length && s[i] == ',') { i++; continue; }
                if (i < s.Length && s[i] == '}') { i++; return map; }
                throw new FormatException("expected , or } in JSON");
            }
        }

        private List<object> ReadArray()
        {
            List<object> list = new List<object>();
            i++;
            Space();
            if (i < s.Length && s[i] == ']') { i++; return list; }
            while (true)
            {
                list.Add(Value());
                Space();
                if (i < s.Length && s[i] == ',') { i++; continue; }
                if (i < s.Length && s[i] == ']') { i++; return list; }
                throw new FormatException("expected , or ] in JSON");
            }
        }

        private string ReadString()
        {
            StringBuilder b = new StringBuilder();
            i++;
            while (i < s.Length)
            {
                char c = s[i++];
                if (c == '"') return b.ToString();
                if (c != '\\') { b.Append(c); continue; }
                if (i >= s.Length) break;
                char e = s[i++];
                switch (e)
                {
                    case '"': b.Append('"'); break;
                    case '\\': b.Append('\\'); break;
                    case '/': b.Append('/'); break;
                    case 'b': b.Append('\b'); break;
                    case 'f': b.Append('\f'); break;
                    case 'n': b.Append('\n'); break;
                    case 'r': b.Append('\r'); break;
                    case 't': b.Append('\t'); break;
                    case 'u':
                        if (i + 4 > s.Length) throw new FormatException("bad \\u escape in JSON");
                        b.Append((char)int.Parse(s.Substring(i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        i += 4;
                        break;
                    default: throw new FormatException("bad escape in JSON");
                }
            }
            throw new FormatException("unterminated string in JSON");
        }

        private double ReadNumber()
        {
            int start = i;
            while (i < s.Length && "+-0123456789.eE".IndexOf(s[i]) >= 0) i++;
            if (start == i) throw new FormatException("unexpected character in JSON");
            return double.Parse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        }
    }

    /** JSON text for replies: every character above 0x7E is escaped, so the reply is ASCII. */
    public static class Json
    {
        public static string Write(object value)
        {
            StringBuilder b = new StringBuilder();
            Append(b, value, 0);
            return b.ToString();
        }

        public static string Quote(string text)
        {
            StringBuilder b = new StringBuilder();
            AppendString(b, text ?? "");
            return b.ToString();
        }

        // PowerShell hands nested values over wrapped in PSObject; its BaseObject is the value. Read through
        // reflection, so this file needs no reference to System.Management.Automation.
        private static object Unwrap(object value)
        {
            for (int n = 0; n < 4 && value != null && value.GetType().FullName == "System.Management.Automation.PSObject"; n++)
            {
                PropertyInfo baseObject = value.GetType().GetProperty("BaseObject");
                if (baseObject == null) break;
                value = baseObject.GetValue(value, null);
            }
            return value;
        }

        private static void Append(StringBuilder b, object value, int depth)
        {
            if (depth > 12) throw new InvalidOperationException("reply nests too deeply");
            value = Unwrap(value);
            if (value == null) { b.Append("null"); return; }
            if (value is string) { AppendString(b, (string)value); return; }
            if (value is char) { AppendString(b, value.ToString()); return; }
            if (value is bool) { b.Append((bool)value ? "true" : "false"); return; }
            if (value is Enum) { AppendString(b, value.ToString()); return; }
            if (value is double || value is float)
            {
                double d = Convert.ToDouble(value, CultureInfo.InvariantCulture);
                if (double.IsNaN(d) || double.IsInfinity(d)) b.Append("null");
                else b.Append(d.ToString("R", CultureInfo.InvariantCulture));
                return;
            }
            if (value is int || value is long || value is short || value is byte || value is uint || value is ulong || value is ushort || value is sbyte || value is decimal)
            {
                b.Append(Convert.ToString(value, CultureInfo.InvariantCulture));
                return;
            }
            IDictionary map = value as IDictionary;
            if (map != null)
            {
                b.Append('{');
                bool first = true;
                foreach (DictionaryEntry entry in map)
                {
                    if (!first) b.Append(',');
                    first = false;
                    AppendString(b, Convert.ToString(Unwrap(entry.Key), CultureInfo.InvariantCulture));
                    b.Append(':');
                    Append(b, entry.Value, depth + 1);
                }
                b.Append('}');
                return;
            }
            IEnumerable list = value as IEnumerable;
            if (list != null)
            {
                b.Append('[');
                bool first = true;
                foreach (object item in list)
                {
                    if (!first) b.Append(',');
                    first = false;
                    Append(b, item, depth + 1);
                }
                b.Append(']');
                return;
            }
            AppendString(b, value.ToString());
        }

        private static void AppendString(StringBuilder b, string text)
        {
            b.Append('"');
            foreach (char c in text)
            {
                if (c == '"') b.Append("\\\"");
                else if (c == '\\') b.Append("\\\\");
                else if (c == '\n') b.Append("\\n");
                else if (c == '\r') b.Append("\\r");
                else if (c == '\t') b.Append("\\t");
                else if (c < 0x20 || c > 0x7E) b.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                else b.Append(c);
            }
            b.Append('"');
        }
    }

    internal static class Native
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

        [StructLayout(LayoutKind.Sequential)]
        public struct POINT { public int X; public int Y; }

        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        public struct MONITORINFO { public int cbSize; public RECT rcMonitor; public RECT rcWork; public uint dwFlags; }

        [StructLayout(LayoutKind.Sequential)]
        public struct MOUSEINPUT { public int dx; public int dy; public uint mouseData; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }

        [StructLayout(LayoutKind.Sequential)]
        public struct KEYBDINPUT { public ushort wVk; public ushort wScan; public uint dwFlags; public uint time; public IntPtr dwExtraInfo; }

        [StructLayout(LayoutKind.Explicit)]
        public struct INPUTUNION { [FieldOffset(0)] public MOUSEINPUT mi; [FieldOffset(0)] public KEYBDINPUT ki; }

        [StructLayout(LayoutKind.Sequential)]
        public struct INPUT { public uint type; public INPUTUNION u; }

        public delegate bool EnumWindowsProc(IntPtr hwnd, IntPtr lParam);
        public delegate bool MonitorEnumProc(IntPtr monitor, IntPtr hdc, ref RECT rect, IntPtr data);

        [DllImport("user32.dll")] public static extern bool SetProcessDpiAwarenessContext(IntPtr value);
        [DllImport("user32.dll")] public static extern IntPtr GetThreadDpiAwarenessContext();
        [DllImport("user32.dll")] public static extern bool AreDpiAwarenessContextsEqual(IntPtr a, IntPtr b);
        [DllImport("user32.dll")] public static extern int GetAwarenessFromDpiAwarenessContext(IntPtr value);
        [DllImport("shcore.dll")] public static extern int SetProcessDpiAwareness(int value);
        [DllImport("user32.dll")] public static extern bool SetProcessDPIAware();

        [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
        [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool IsIconic(IntPtr hwnd);
        [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hwnd, int command);
        [DllImport("user32.dll")] public static extern IntPtr GetAncestor(IntPtr hwnd, uint flags);
        [DllImport("user32.dll")] public static extern IntPtr GetWindow(IntPtr hwnd, uint command);
        [DllImport("user32.dll")] public static extern int GetWindowLong(IntPtr hwnd, int index);
        [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr hwnd, out uint pid);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetWindowText(IntPtr hwnd, StringBuilder text, int max);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern int GetClassName(IntPtr hwnd, StringBuilder text, int max);
        [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hwnd, out RECT rect);
        [DllImport("user32.dll")] public static extern bool EnumWindows(EnumWindowsProc proc, IntPtr lParam);
        [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr parent, EnumWindowsProc proc, IntPtr lParam);
        [DllImport("user32.dll")] public static extern bool PostMessage(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern IntPtr SendMessageTimeout(IntPtr hwnd, uint msg, IntPtr wParam, string lParam, uint flags, uint timeout, out IntPtr result);
        [DllImport("user32.dll")] public static extern bool GetCursorPos(out POINT point);
        [DllImport("user32.dll", SetLastError = true)] public static extern uint SendInput(uint count, INPUT[] inputs, int size);
        [DllImport("user32.dll")] public static extern int GetSystemMetrics(int index);
        [DllImport("user32.dll")] public static extern uint MapVirtualKey(uint code, uint mapType);
        [DllImport("user32.dll")] public static extern bool LockWorkStation();
        [DllImport("user32.dll")] public static extern bool EnumDisplayMonitors(IntPtr hdc, IntPtr clip, MonitorEnumProc proc, IntPtr data);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] public static extern bool GetMonitorInfo(IntPtr monitor, ref MONITORINFO info);
        [DllImport("user32.dll", SetLastError = true)] public static extern IntPtr OpenInputDesktop(uint flags, bool inherit, uint access);
        [DllImport("user32.dll")] public static extern bool CloseDesktop(IntPtr desktop);
        [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)] public static extern bool GetUserObjectInformation(IntPtr obj, int index, StringBuilder info, int length, out int needed);

        [DllImport("dwmapi.dll")] public static extern int DwmGetWindowAttribute(IntPtr hwnd, int attribute, out RECT value, int size);
        [DllImport("dwmapi.dll", EntryPoint = "DwmGetWindowAttribute")] public static extern int DwmGetWindowAttributeInt(IntPtr hwnd, int attribute, out int value, int size);

        [DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
        [DllImport("kernel32.dll")] public static extern bool CloseHandle(IntPtr handle);
        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)] public static extern bool QueryFullProcessImageName(IntPtr process, uint flags, StringBuilder name, ref int size);
        [DllImport("kernel32.dll")] public static extern bool ProcessIdToSessionId(uint pid, out uint session);
        [DllImport("advapi32.dll", SetLastError = true)] public static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
        [DllImport("advapi32.dll", SetLastError = true)] public static extern bool GetTokenInformation(IntPtr token, int infoClass, IntPtr info, int length, out int needed);
        [DllImport("advapi32.dll")] public static extern IntPtr GetSidSubAuthority(IntPtr sid, uint index);
        [DllImport("advapi32.dll")] public static extern IntPtr GetSidSubAuthorityCount(IntPtr sid);
    }

    // Core Audio, declared by hand (vtable order matters: every method up to the last one used is listed).
    [ComImport, Guid("BCDE0395-E52F-467C-8E3D-C4579291692E")]
    internal class MMDeviceEnumeratorCom { }

    [ComImport, Guid("A95664D2-9614-4F35-A746-DE8DB63617E6"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IMMDeviceEnumerator
    {
        int EnumAudioEndpoints(int dataFlow, int stateMask, out IntPtr devices);
        int GetDefaultAudioEndpoint(int dataFlow, int role, out IMMDevice device);
    }

    [ComImport, Guid("D666063F-1587-4E43-81F1-B948E807363F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IMMDevice
    {
        int Activate(ref Guid iid, int context, IntPtr parameters, [MarshalAs(UnmanagedType.IUnknown)] out object instance);
    }

    [ComImport, Guid("5CDF2C82-841E-4546-9722-0CF74078229A"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IAudioEndpointVolume
    {
        int RegisterControlChangeNotify(IntPtr notify);
        int UnregisterControlChangeNotify(IntPtr notify);
        int GetChannelCount(out uint count);
        int SetMasterVolumeLevel(float level, IntPtr context);
        int SetMasterVolumeLevelScalar(float level, IntPtr context);
        int GetMasterVolumeLevel(out float level);
        int GetMasterVolumeLevelScalar(out float level);
        int SetChannelVolumeLevel(uint channel, float level, IntPtr context);
        int SetChannelVolumeLevelScalar(uint channel, float level, IntPtr context);
        int GetChannelVolumeLevel(uint channel, out float level);
        int GetChannelVolumeLevelScalar(uint channel, out float level);
        int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, IntPtr context);
        int GetMute([MarshalAs(UnmanagedType.Bool)] out bool mute);
    }

    [ComImport, Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IAudioSessionManager2
    {
        int GetAudioSessionControl(IntPtr group, int flags, out IntPtr control);
        int GetSimpleAudioVolume(IntPtr group, int flags, out IntPtr volume);
        int GetSessionEnumerator(out IAudioSessionEnumerator sessions);
    }

    [ComImport, Guid("E2F5BB11-0570-40CA-ACDD-3AA01277DEE8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IAudioSessionEnumerator
    {
        int GetCount(out int count);
        int GetSession(int index, [MarshalAs(UnmanagedType.IUnknown)] out object session);
    }

    [ComImport, Guid("bfb7ff88-7239-4fc9-8fa2-07c950be9c6d"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface IAudioSessionControl2
    {
        int GetState(out int state);
        int GetDisplayName(out IntPtr name);
        int SetDisplayName([MarshalAs(UnmanagedType.LPWStr)] string name, IntPtr context);
        int GetIconPath(out IntPtr path);
        int SetIconPath([MarshalAs(UnmanagedType.LPWStr)] string path, IntPtr context);
        int GetGroupingParam(out Guid group);
        int SetGroupingParam(ref Guid group, IntPtr context);
        int RegisterAudioSessionNotification(IntPtr events);
        int UnregisterAudioSessionNotification(IntPtr events);
        int GetSessionIdentifier(out IntPtr id);
        int GetSessionInstanceIdentifier(out IntPtr id);
        int GetProcessId(out uint pid);
    }

    [ComImport, Guid("87CE5498-68D6-44E5-9215-6DA47EF883D8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    internal interface ISimpleAudioVolume
    {
        int SetMasterVolume(float level, IntPtr context);
        int GetMasterVolume(out float level);
        int SetMute([MarshalAs(UnmanagedType.Bool)] bool mute, IntPtr context);
        int GetMute([MarshalAs(UnmanagedType.Bool)] out bool mute);
    }

    internal sealed class Found
    {
        public string Ref;
        public string Role;
        public string Label;
        public string Value;
        public bool Password;
        public bool Focused;
        public System.Windows.Rect Bounds;
    }

    internal sealed class Node
    {
        public AutomationElement El;
        public string Path;
        public int Depth;
        public Node(AutomationElement el, string path, int depth) { El = el; Path = path; Depth = depth; }
    }

    public static class Lr
    {
        private const int MaxNodes = 2500;
        private const int MaxControls = 150;
        private const int MaxText = 1500;
        private const int MaxDepth = 25;
        private const uint QueryLimited = 0x1000;
        private const int SecurityHigh = 0x3000;
        private static string dpi = "unknown";

        // ---- start ----

        /** Per-monitor DPI awareness first (see the file comment), then who and what this worker is. */
        public static string Hello()
        {
            dpi = MakeDpiAware();
            Dictionary<string, object> hello = new Dictionary<string, object>();
            hello["pid"] = Process.GetCurrentProcess().Id;
            hello["dpi"] = dpi;
            return Json.Write(hello);
        }

        private static string MakeDpiAware()
        {
            IntPtr v2 = new IntPtr(-4);
            try
            {
                Native.SetProcessDpiAwarenessContext(v2);
                if (Native.AreDpiAwarenessContextsEqual(Native.GetThreadDpiAwarenessContext(), v2)) return "per_monitor_v2";
                int awareness = Native.GetAwarenessFromDpiAwarenessContext(Native.GetThreadDpiAwarenessContext());
                return awareness == 2 ? "per_monitor" : awareness == 1 ? "system" : "unaware";
            }
            catch (EntryPointNotFoundException) { }
            try
            {
                if (Native.SetProcessDpiAwareness(2) == 0) return "per_monitor";
            }
            catch (DllNotFoundException) { }
            catch (EntryPointNotFoundException) { }
            return Native.SetProcessDPIAware() ? "system" : "unaware";
        }

        public static Request Parse(string line)
        {
            Dictionary<string, object> raw = new JsonReader(line).Document() as Dictionary<string, object>;
            if (raw == null) throw new FormatException("a request should be a JSON object");
            Request request = new Request();
            object id, op, args;
            if (!raw.TryGetValue("id", out id) || !(id is double)) throw new FormatException("a request needs a numeric id");
            request.Id = (long)(double)id;
            if (!raw.TryGetValue("op", out op) || !(op is string)) throw new FormatException("a request needs an op");
            request.Op = (string)op;
            request.Args = raw.TryGetValue("args", out args) && args is Dictionary<string, object> ? (Dictionary<string, object>)args : new Dictionary<string, object>();
            return request;
        }

        public static string Run(string op, Dictionary<string, object> a)
        {
            switch (op)
            {
                case "observe": return Json.Write(Observe(a));
                case "semantic": return Json.Write(Semantic(a));
                case "point": return Json.Write(PointFor(a));
                case "click": return Json.Write(Click(a));
                case "select_all": return Json.Write(SelectAll(a));
                case "type": return Json.Write(TypeText(a));
                case "key": return Json.Write(Key(a));
                case "wheel": return Json.Write(Wheel(a));
                case "pointer": return Json.Write(Pointer());
                case "displays": return Json.Write(Displays());
                case "screenshot": return Json.Write(Screenshot(a));
                case "windows": return Json.Write(Windows());
                case "foreground": return Json.Write(Foreground(a));
                case "close": return Json.Write(Close(a));
                case "tabs": return Json.Write(Tabs());
                case "tab_select": return Json.Write(TabSelect(a));
                case "volume_read": return Json.Write(VolumeRead(a));
                case "volume_write": return Json.Write(VolumeWrite(a));
                case "dark_mode": return Json.Write(DarkMode(a));
                case "lock": return Json.Write(Lock());
                case "media_key": return Json.Write(MediaKey(a));
                case "access": return Json.Write(Access());
                default: throw new LrError("UNKNOWN_OP", op);
            }
        }

        // ---- small helpers ----

        private static Dictionary<string, object> D() { return new Dictionary<string, object>(); }

        private static string Str(Dictionary<string, object> a, string key)
        {
            object v;
            return a.TryGetValue(key, out v) && v is string ? (string)v : null;
        }

        private static double Num(Dictionary<string, object> a, string key, double fallback)
        {
            object v;
            return a.TryGetValue(key, out v) && v is double ? (double)v : fallback;
        }

        private static bool Has(Dictionary<string, object> a, string key)
        {
            object v;
            return a.TryGetValue(key, out v) && v != null;
        }

        private static IntPtr Hwnd(Dictionary<string, object> a)
        {
            return new IntPtr((long)Num(a, "hwnd", 0));
        }

        private static Dictionary<string, object> RectJson(double x, double y, double w, double h)
        {
            Dictionary<string, object> r = D();
            r["x"] = (int)Math.Round(x);
            r["y"] = (int)Math.Round(y);
            r["w"] = (int)Math.Round(w);
            r["h"] = (int)Math.Round(h);
            return r;
        }

        private static string Clip(string text, int max)
        {
            if (text == null) return "";
            string flat = Regex.Replace(text, "\\s+", " ").Trim();
            return flat.Length > max ? flat.Substring(0, max - 1) + "…" : flat;
        }

        private static string WindowText(IntPtr hwnd)
        {
            StringBuilder b = new StringBuilder(512);
            Native.GetWindowText(hwnd, b, b.Capacity);
            return b.ToString();
        }

        private static string ClassOf(IntPtr hwnd)
        {
            StringBuilder b = new StringBuilder(256);
            Native.GetClassName(hwnd, b, b.Capacity);
            return b.ToString();
        }

        private static IntPtr Root(IntPtr hwnd) { return hwnd == IntPtr.Zero ? hwnd : Native.GetAncestor(hwnd, 2); }

        // ---- processes, apps, integrity ----

        private static string ImagePath(uint pid)
        {
            IntPtr process = Native.OpenProcess(QueryLimited, false, pid);
            if (process == IntPtr.Zero) return null;
            try
            {
                StringBuilder b = new StringBuilder(1024);
                int size = b.Capacity;
                return Native.QueryFullProcessImageName(process, 0, b, ref size) ? b.ToString(0, size) : null;
            }
            finally { Native.CloseHandle(process); }
        }

        /** The executable's base name ("chrome"), for matching what people type. */
        private static string ProcessName(uint pid)
        {
            string path = ImagePath(pid);
            if (path != null) return Path.GetFileNameWithoutExtension(path);
            try { return Process.GetProcessById((int)pid).ProcessName; }
            catch (ArgumentException) { return ""; }
            catch (InvalidOperationException) { return ""; }
        }

        /** The app's own name ("Google Chrome"): the executable's file description, else its base name. */
        private static string AppName(uint pid)
        {
            string path = ImagePath(pid);
            if (path != null)
            {
                try
                {
                    string description = FileVersionInfo.GetVersionInfo(path).FileDescription;
                    if (!string.IsNullOrEmpty(description) && description.Trim().Length > 0) return description.Trim();
                }
                catch (FileNotFoundException) { }
                return Path.GetFileNameWithoutExtension(path);
            }
            return ProcessName(pid);
        }

        /**
         * The process behind a top-level window. Store apps (Settings, Calculator) draw inside an
         * ApplicationFrameWindow owned by ApplicationFrameHost; the app itself is the child window of another process.
         */
        private static uint OwnerPid(IntPtr hwnd)
        {
            uint pid;
            Native.GetWindowThreadProcessId(hwnd, out pid);
            if (ClassOf(hwnd) != "ApplicationFrameWindow") return pid;
            uint frame = pid, inner = 0;
            Native.EnumChildWindows(hwnd, delegate (IntPtr child, IntPtr unused)
            {
                uint childPid;
                Native.GetWindowThreadProcessId(child, out childPid);
                if (childPid != frame) { inner = childPid; return false; }
                return true;
            }, IntPtr.Zero);
            return inner != 0 ? inner : pid;
        }

        /** The mandatory integrity level's RID (0x2000 medium, 0x3000 high), or -1 when the token cannot be read. */
        private static int Integrity(uint pid)
        {
            IntPtr process = Native.OpenProcess(QueryLimited, false, pid);
            if (process == IntPtr.Zero) return -1;
            try
            {
                IntPtr token;
                if (!Native.OpenProcessToken(process, 0x0008, out token)) return -1;
                try
                {
                    int needed;
                    Native.GetTokenInformation(token, 25, IntPtr.Zero, 0, out needed);
                    if (needed <= 0) return -1;
                    IntPtr buffer = Marshal.AllocHGlobal(needed);
                    try
                    {
                        if (!Native.GetTokenInformation(token, 25, buffer, needed, out needed)) return -1;
                        IntPtr sid = Marshal.ReadIntPtr(buffer);
                        int count = Marshal.ReadByte(Native.GetSidSubAuthorityCount(sid));
                        return Marshal.ReadInt32(Native.GetSidSubAuthority(sid, (uint)(count - 1)));
                    }
                    finally { Marshal.FreeHGlobal(buffer); }
                }
                finally { Native.CloseHandle(token); }
            }
            finally { Native.CloseHandle(process); }
        }

        private static int OwnIntegrity() { return Integrity((uint)Process.GetCurrentProcess().Id); }

        /**
         * UIPI: Windows silently drops input (and most UI Automation) sent to a process of higher integrity, such as an
         * app run as administrator, unless this one is at least as high. A token that cannot be read is an elevated
         * one: a normal process can read the token of another normal process of the same user.
         */
        private static bool Elevated(uint pid)
        {
            int own = OwnIntegrity();
            int theirs = Integrity(pid);
            if (theirs < 0) return own >= 0 && own < SecurityHigh;
            return own >= 0 && theirs > own;
        }

        /** The input desktop is not the user's (a UAC prompt, the lock or sign-in screen): nothing can read or drive it. */
        private static bool SecureDesktop()
        {
            IntPtr desktop = Native.OpenInputDesktop(0, false, 0x0001);
            if (desktop == IntPtr.Zero) return true;
            try
            {
                StringBuilder name = new StringBuilder(256);
                int needed;
                if (!Native.GetUserObjectInformation(desktop, 2, name, name.Capacity * 2, out needed)) return false;
                return !string.Equals(name.ToString(), "Default", StringComparison.OrdinalIgnoreCase);
            }
            finally { Native.CloseDesktop(desktop); }
        }

        private static void NotSecure()
        {
            if (SecureDesktop()) throw new LrError("SECURE_DESKTOP", "");
        }

        private static IntPtr FrontWindow()
        {
            NotSecure();
            IntPtr hwnd = Root(Native.GetForegroundWindow());
            if (hwnd == IntPtr.Zero) throw new LrError("NO_APP", "");
            return hwnd;
        }

        private static void NotElevated(IntPtr hwnd)
        {
            uint pid = OwnerPid(hwnd);
            if (Elevated(pid)) throw new LrError("ELEVATED", AppName(pid));
        }

        /** The window's visible rectangle: DWM's extended frame bounds have no invisible resize border or shadow. */
        private static Native.RECT WindowRect(IntPtr hwnd)
        {
            Native.RECT r;
            if (Native.DwmGetWindowAttribute(hwnd, 9, out r, Marshal.SizeOf(typeof(Native.RECT))) != 0) Native.GetWindowRect(hwnd, out r);
            return r;
        }

        // ---- observe ----

        private static readonly Regex AddressBar = new Regex("address and search bar|search or enter address|search with .+ or enter address|^address bar$|search or type (a )?(web address|url)", RegexOptions.IgnoreCase);
        private static readonly string[] Browsers = new string[] { "chrome", "msedge", "firefox", "brave", "opera", "vivaldi", "chromium" };

        private static bool IsBrowser(string process)
        {
            foreach (string name in Browsers) if (string.Equals(process, name, StringComparison.OrdinalIgnoreCase)) return true;
            return false;
        }

        private static CacheRequest ObserveCache()
        {
            CacheRequest cache = new CacheRequest();
            cache.TreeFilter = Automation.ControlViewCondition;
            cache.TreeScope = TreeScope.Element;
            cache.Add(AutomationElement.NameProperty);
            cache.Add(AutomationElement.HelpTextProperty);
            cache.Add(AutomationElement.ControlTypeProperty);
            cache.Add(AutomationElement.BoundingRectangleProperty);
            cache.Add(AutomationElement.IsOffscreenProperty);
            cache.Add(AutomationElement.IsEnabledProperty);
            cache.Add(AutomationElement.IsPasswordProperty);
            cache.Add(AutomationElement.HasKeyboardFocusProperty);
            cache.Add(AutomationElement.IsValuePatternAvailableProperty);
            cache.Add(AutomationElement.RuntimeIdProperty);
            cache.Add(ValuePattern.IsReadOnlyProperty);
            return cache;
        }

        private static bool CachedBool(AutomationElement el, AutomationProperty property)
        {
            object v = el.GetCachedPropertyValue(property, true);
            return v is bool && (bool)v;
        }

        private static string RuntimeId(int[] id)
        {
            if (id == null || id.Length == 0) return "";
            StringBuilder b = new StringBuilder();
            for (int n = 0; n < id.Length; n++)
            {
                if (n > 0) b.Append('.');
                b.Append(id[n].ToString(CultureInfo.InvariantCulture));
            }
            return b.ToString();
        }

        private static string RoleOf(ControlType type, bool editable)
        {
            if (type == ControlType.Button || type == ControlType.SplitButton) return "button";
            if (type == ControlType.Hyperlink) return "link";
            if (type == ControlType.Edit) return "textbox";
            if (type == ControlType.Document) return editable ? "textbox" : null;
            if (type == ControlType.ComboBox) return editable ? "textbox" : "button";
            if (type == ControlType.TabItem) return "tab";
            if (type == ControlType.ListItem || type == ControlType.DataItem || type == ControlType.TreeItem) return "row";
            if (type == ControlType.MenuItem) return "menuitem";
            if (type == ControlType.CheckBox || type == ControlType.RadioButton) return "checkbox";
            return null;
        }

        /** Controls whose children are not worth the walk's budget: their own name is their label. */
        private static bool Leaf(ControlType type)
        {
            return type == ControlType.Text || type == ControlType.Image || type == ControlType.Edit || type == ControlType.Button ||
                type == ControlType.SplitButton || type == ControlType.CheckBox || type == ControlType.RadioButton ||
                type == ControlType.ScrollBar || type == ControlType.Hyperlink || type == ControlType.ProgressBar ||
                type == ControlType.Slider || type == ControlType.Spinner || type == ControlType.Separator ||
                type == ControlType.Thumb || type == ControlType.ComboBox || type == ControlType.MenuItem;
        }

        private static Dictionary<string, object> Observe(Dictionary<string, object> a)
        {
            IntPtr hwnd = FrontWindow();
            uint pid = OwnerPid(hwnd);
            string app = AppName(pid);
            if (Elevated(pid)) throw new LrError("ELEVATED", app);
            bool browser = IsBrowser(ProcessName(pid));
            int budgetMs = (int)Num(a, "budgetMs", 3000);
            Native.RECT wr = WindowRect(hwnd);
            CacheRequest cache = ObserveCache();
            AutomationElement win;
            try
            {
                using (cache.Activate()) { win = AutomationElement.FromHandle(hwnd); }
            }
            catch (ElementNotAvailableException) { throw new LrError("NO_WINDOW", app); }

            string title = WindowText(hwnd);
            if (title.Length == 0) { try { title = win.Cached.Name ?? ""; } catch (InvalidOperationException) { } }

            List<Found> found = new List<Found>();
            StringBuilder text = new StringBuilder();
            string url = null;
            Queue<Node> queue = new Queue<Node>();
            queue.Enqueue(new Node(win, "", 0));
            Stopwatch clock = Stopwatch.StartNew();
            int visited = 0;
            bool truncated = false;
            string prefix = ((long)hwnd).ToString(CultureInfo.InvariantCulture) + ":";

            while (queue.Count > 0 && !truncated)
            {
                Node node = queue.Dequeue();
                AutomationElementCollection kids;
                try
                {
                    using (cache.Activate()) { kids = node.El.FindAll(TreeScope.Children, Automation.ControlViewCondition); }
                }
                catch (ElementNotAvailableException) { continue; }
                catch (COMException) { continue; }
                catch (InvalidOperationException) { continue; }
                for (int i = 0; i < kids.Count; i++)
                {
                    if (visited >= MaxNodes || found.Count >= MaxControls || clock.ElapsedMilliseconds > budgetMs) { truncated = true; break; }
                    visited++;
                    AutomationElement el = kids[i];
                    ControlType type;
                    System.Windows.Rect r;
                    bool offscreen, enabled;
                    try
                    {
                        type = el.Cached.ControlType;
                        r = el.Cached.BoundingRectangle;
                        offscreen = el.Cached.IsOffscreen;
                        enabled = el.Cached.IsEnabled;
                    }
                    catch (ElementNotAvailableException) { continue; }
                    catch (InvalidOperationException) { continue; }
                    bool sized = !r.IsEmpty && r.Width > 0 && r.Height > 0;
                    if (sized && (r.Left >= wr.Right || r.Top >= wr.Bottom || r.Right <= wr.Left || r.Bottom <= wr.Top)) continue;
                    if (sized && offscreen) continue;
                    string path = node.Path.Length == 0 ? i.ToString(CultureInfo.InvariantCulture) : node.Path + "." + i.ToString(CultureInfo.InvariantCulture);
                    string name = "";
                    try { name = el.Cached.Name ?? ""; } catch (InvalidOperationException) { }

                    if (type == ControlType.Text)
                    {
                        if (text.Length < MaxText && name.Length > 0) text.Append(text.Length > 0 ? " " : "").Append(Clip(name, MaxText));
                    }
                    else if (sized && !offscreen && enabled)
                    {
                        bool valued = CachedBool(el, AutomationElement.IsValuePatternAvailableProperty);
                        bool readOnly = valued && CachedBool(el, ValuePattern.IsReadOnlyProperty);
                        bool password = CachedBool(el, AutomationElement.IsPasswordProperty);
                        string role = RoleOf(type, valued && !readOnly);
                        if (role == "textbox" && type == ControlType.Edit && valued && readOnly && !password)
                        {
                            // A read-only edit is text on the page (a label, a status line), not a field to type into.
                            string shown = ReadValue(el);
                            if (text.Length < MaxText && shown.Length > 0) text.Append(text.Length > 0 ? " " : "").Append(Clip(shown, MaxText));
                            role = null;
                        }
                        if (role != null)
                        {
                            string label = name;
                            if (label.Length == 0) { try { label = el.Cached.HelpText ?? ""; } catch (InvalidOperationException) { } }
                            label = Clip(label, 100);
                            if (role != "row" || label.Length > 0)
                            {
                                Found f = new Found();
                                object rid = el.GetCachedPropertyValue(AutomationElement.RuntimeIdProperty, true);
                                string id = RuntimeId(rid as int[]);
                                f.Ref = prefix + path + (id.Length > 0 ? "#" + id : "");
                                f.Role = role;
                                f.Label = label;
                                f.Bounds = r;
                                f.Password = password;
                                f.Focused = CachedBool(el, AutomationElement.HasKeyboardFocusProperty);
                                // A password field's text is never read, not even to be dropped later.
                                if (role == "textbox" && !password) f.Value = Clip(ReadValue(el), 200);
                                if (browser && url == null && role == "textbox" && !password && !f.Focused && AddressBar.IsMatch(name)) url = ReadValue(el).Trim();
                                found.Add(f);
                            }
                        }
                    }
                    if (!Leaf(type) && node.Depth + 1 < MaxDepth) queue.Enqueue(new Node(el, path, node.Depth + 1));
                }
                if (!truncated && queue.Count > 0 && (visited >= MaxNodes || found.Count >= MaxControls)) truncated = true;
            }

            found.Sort(delegate (Found x, Found y)
            {
                int row = Math.Round(x.Bounds.Top / 8).CompareTo(Math.Round(y.Bounds.Top / 8));
                return row != 0 ? row : x.Bounds.Left.CompareTo(y.Bounds.Left);
            });
            List<object> controls = new List<object>();
            foreach (Found f in found)
            {
                Dictionary<string, object> c = D();
                c["ref"] = f.Ref;
                c["role"] = f.Role;
                c["label"] = f.Label;
                if (f.Role == "textbox")
                {
                    if (f.Password) c["password"] = true;
                    else c["value"] = f.Value ?? "";
                    if (f.Focused) c["focused"] = true;
                }
                c["bounds"] = RectJson(f.Bounds.Left, f.Bounds.Top, f.Bounds.Width, f.Bounds.Height);
                controls.Add(c);
            }
            Dictionary<string, object> o = D();
            o["app"] = app;
            o["title"] = title;
            if (url != null && url.Length > 0) o["url"] = url;
            o["controls"] = controls;
            string excerpt = text.ToString();
            o["text"] = excerpt.Length > MaxText ? excerpt.Substring(0, MaxText) : excerpt;
            o["window"] = RectJson(wr.Left, wr.Top, wr.Right - wr.Left, wr.Bottom - wr.Top);
            o["truncated"] = truncated;
            return o;
        }

        private static string ReadValue(AutomationElement el)
        {
            try
            {
                object p;
                if (!el.TryGetCurrentPattern(ValuePattern.Pattern, out p)) return "";
                return ((ValuePattern)p).Current.Value ?? "";
            }
            catch (ElementNotAvailableException) { return ""; }
            catch (InvalidOperationException) { return ""; }
            catch (COMException) { return ""; }
        }

        // ---- refs ----

        /**
         * A ref is "<hwnd>:<child-index path in the ControlView>#<RuntimeId>". The path is walked again from the window,
         * so the action reaches the control where it is now; the RuntimeId, when the provider gave one, must still
         * match, so a list that changed under the path never lets the action land on a neighbour.
         */
        private static AutomationElement Resolve(string reference, out IntPtr hwnd)
        {
            Match m = Regex.Match(reference ?? "", "^(\\d+):(\\d+(?:\\.\\d+)*)(?:#(-?\\d+(?:\\.-?\\d+)*))?$");
            if (!m.Success) throw new LrError("GONE", "");
            hwnd = new IntPtr(long.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture));
            if (!Native.IsWindow(hwnd)) throw new LrError("GONE", "");
            AutomationElement el;
            try
            {
                el = AutomationElement.FromHandle(hwnd);
                foreach (string part in m.Groups[2].Value.Split('.'))
                {
                    int index = int.Parse(part, CultureInfo.InvariantCulture);
                    AutomationElementCollection kids = el.FindAll(TreeScope.Children, Automation.ControlViewCondition);
                    if (index >= kids.Count) throw new LrError("GONE", "");
                    el = kids[index];
                }
                if (m.Groups[3].Success && RuntimeId(el.GetRuntimeId()) != m.Groups[3].Value) throw new LrError("GONE", "");
            }
            catch (ElementNotAvailableException) { throw new LrError("GONE", ""); }
            return el;
        }

        // ---- semantic actions (UI Automation patterns: no pointer, no keys) ----

        private static Dictionary<string, object> Done(bool done, string how)
        {
            Dictionary<string, object> o = D();
            o["done"] = done;
            o["how"] = how;
            return o;
        }

        private static Dictionary<string, object> Semantic(Dictionary<string, object> a)
        {
            NotSecure();
            string kind = Str(a, "kind");
            IntPtr hwnd;
            AutomationElement el;
            if (kind == "scroll" && !Has(a, "ref"))
            {
                hwnd = FrontWindow();
                try { el = AutomationElement.FromHandle(hwnd); }
                catch (ElementNotAvailableException) { throw new LrError("NO_WINDOW", ""); }
            }
            else el = Resolve(Str(a, "ref"), out hwnd);
            NotElevated(hwnd);
            try
            {
                if (kind == "click") return SemanticClick(el);
                if (kind == "type") return SemanticType(el, Str(a, "text") ?? "", Str(a, "label") ?? "");
                if (kind == "scroll") return SemanticScroll(el, Str(a, "direction"), !Has(a, "ref"));
            }
            catch (ElementNotAvailableException) { throw new LrError("GONE", ""); }
            throw new LrError("UNKNOWN_OP", kind);
        }

        /** How long a click's pattern call may take before it counts as done and still running. */
        private const int PatternWaitMs = 2500;

        /**
         * Runs a pattern call on its own thread and waits up to PatternWaitMs. A button that opens a modal dialog can
         * keep Invoke (and Toggle, Select, Expand) from returning until the dialog closes: the click did happen, so it
         * is reported done with "pending", and the thread finishes on its own when the dialog goes, without holding up
         * the worker or getting it killed. An exception thrown within the wait is rethrown, so the step still fails.
         */
        private static Dictionary<string, object> Pattern(ThreadStart work, string how)
        {
            Exception failure = null;
            Thread thread = new Thread(delegate ()
            {
                try { work(); }
                catch (Exception error) { failure = error; }
            });
            thread.IsBackground = true;
            thread.Start();
            bool finished = thread.Join(PatternWaitMs);
            if (finished && failure != null) throw failure;
            Dictionary<string, object> done = Done(true, how);
            if (!finished) done["pending"] = true;
            return done;
        }

        private static Dictionary<string, object> SemanticClick(AutomationElement el)
        {
            object p;
            try
            {
                if (el.TryGetCurrentPattern(InvokePattern.Pattern, out p))
                {
                    InvokePattern invoke = (InvokePattern)p;
                    return Pattern(delegate () { invoke.Invoke(); }, "invoke");
                }
                if (el.TryGetCurrentPattern(SelectionItemPattern.Pattern, out p))
                {
                    SelectionItemPattern select = (SelectionItemPattern)p;
                    return Pattern(delegate () { select.Select(); }, "select");
                }
                if (el.TryGetCurrentPattern(TogglePattern.Pattern, out p))
                {
                    TogglePattern toggle = (TogglePattern)p;
                    return Pattern(delegate () { toggle.Toggle(); }, "toggle");
                }
                if (el.TryGetCurrentPattern(ExpandCollapsePattern.Pattern, out p))
                {
                    ExpandCollapsePattern ec = (ExpandCollapsePattern)p;
                    return Pattern(delegate ()
                    {
                        if (ec.Current.ExpandCollapseState == ExpandCollapseState.Collapsed) ec.Expand(); else ec.Collapse();
                    }, "expand");
                }
                ControlType type = el.Current.ControlType;
                if (type == ControlType.Edit || type == ControlType.Document || type == ControlType.ComboBox)
                {
                    // Clicking into a field only focuses it.
                    el.SetFocus();
                    return Done(true, "focus");
                }
            }
            catch (InvalidOperationException) { }
            catch (COMException) { }
            return Done(false, "");
        }

        private static string Lines(string text) { return (text ?? "").Replace("\r\n", "\n").Replace('\r', '\n'); }

        private static Dictionary<string, object> SemanticType(AutomationElement el, string text, string label)
        {
            object p;
            if (el.Current.IsPassword) return Done(false, "password");
            if (!el.TryGetCurrentPattern(ValuePattern.Pattern, out p)) return Done(false, "");
            ValuePattern vp = (ValuePattern)p;
            if (vp.Current.IsReadOnly) return Done(false, "read_only");
            try { vp.SetValue(text); }
            // InvalidOperationException covers ElementNotEnabledException and a provider that refuses the value.
            catch (InvalidOperationException) { return Done(false, ""); }
            catch (COMException) { return Done(false, ""); }
            string now = ReadValue(el);
            if (Lines(now) != Lines(text)) throw new LrError("DID_NOT_TAKE", label);
            return Done(true, "value");
        }

        private static ScrollPattern Scrollable(AutomationElement el, bool whole)
        {
            TreeWalker walker = TreeWalker.ControlViewWalker;
            AutomationElement at = el;
            for (int n = 0; at != null && n < 30; n++)
            {
                object p;
                if (at.TryGetCurrentPattern(ScrollPattern.Pattern, out p) && ((ScrollPattern)p).Current.VerticallyScrollable) return (ScrollPattern)p;
                if (whole) break;
                at = walker.GetParent(at);
                if (at == null || at == AutomationElement.RootElement) break;
            }
            if (!whole) return null;
            // No ref: the first part of the window that scrolls vertically (a page, a list).
            Condition scrolls = new AndCondition(
                new PropertyCondition(AutomationElement.IsScrollPatternAvailableProperty, true),
                new PropertyCondition(ScrollPattern.VerticallyScrollableProperty, true));
            AutomationElement inner = el.FindFirst(TreeScope.Descendants, scrolls);
            object q;
            return inner != null && inner.TryGetCurrentPattern(ScrollPattern.Pattern, out q) ? (ScrollPattern)q : null;
        }

        private static Dictionary<string, object> SemanticScroll(AutomationElement el, string direction, bool whole)
        {
            ScrollPattern sp = Scrollable(el, whole);
            if (sp == null) return Done(false, "");
            try
            {
                if (direction == "top") sp.SetScrollPercent(ScrollPattern.NoScroll, 0);
                else if (direction == "bottom") sp.SetScrollPercent(ScrollPattern.NoScroll, 100);
                else if (direction == "up") sp.ScrollVertical(ScrollAmount.LargeDecrement);
                else if (direction == "down") sp.ScrollVertical(ScrollAmount.LargeIncrement);
                else throw new LrError("UNKNOWN_OP", direction);
            }
            catch (InvalidOperationException)
            {
                // Already at that end: nothing to scroll is still done.
                return Done(true, "edge");
            }
            return Done(true, "scroll");
        }

        // ---- pointer and keyboard (SendInput) ----

        /**
         * Brings the window to the front for pointer or key input, without tricks: Windows lets a process take the
         * foreground only under its foreground lock rules (it is the foreground process, was started by it, received
         * the last input, and so on). When it says no, the action fails with a sentence instead of typing into
         * whatever is in front.
         */
        private static void Front(IntPtr hwnd)
        {
            hwnd = Root(hwnd);
            if (Root(Native.GetForegroundWindow()) == hwnd) return;
            if (Native.IsIconic(hwnd)) Native.ShowWindow(hwnd, 9);
            Native.SetForegroundWindow(hwnd);
            for (int n = 0; n < 10 && Root(Native.GetForegroundWindow()) != hwnd; n++) Thread.Sleep(40);
            if (Root(Native.GetForegroundWindow()) != hwnd) throw new LrError("NOT_FRONT", AppName(OwnerPid(hwnd)));
        }

        /** Before any synthetic input: the user's desktop, the same window still in front, and not an elevated one. */
        private static void InputGuard(IntPtr hwnd)
        {
            NotSecure();
            IntPtr front = Root(Native.GetForegroundWindow());
            if (front == IntPtr.Zero) throw new LrError("NO_APP", "");
            if (hwnd != IntPtr.Zero && front != Root(hwnd)) throw new LrError("FRONT_CHANGED", "");
            NotElevated(front);
        }

        private static Dictionary<string, object> PointFor(Dictionary<string, object> a)
        {
            NotSecure();
            IntPtr hwnd;
            double x, y;
            if (Has(a, "ref"))
            {
                AutomationElement el = Resolve(Str(a, "ref"), out hwnd);
                NotElevated(hwnd);
                System.Windows.Rect r;
                try { r = el.Current.BoundingRectangle; }
                catch (ElementNotAvailableException) { throw new LrError("GONE", ""); }
                if (r.IsEmpty || r.Width <= 0 || r.Height <= 0) throw new LrError("GONE", "");
                System.Windows.Point clickable;
                bool has = false;
                try { has = el.TryGetClickablePoint(out clickable); }
                catch (ElementNotAvailableException) { clickable = new System.Windows.Point(); }
                if (has && r.Contains(clickable)) { x = clickable.X; y = clickable.Y; }
                else { x = r.Left + r.Width / 2; y = r.Top + r.Height / 2; }
            }
            else
            {
                hwnd = FrontWindow();
                NotElevated(hwnd);
                Native.RECT wr = WindowRect(hwnd);
                if (wr.Right <= wr.Left || wr.Bottom <= wr.Top) throw new LrError("NO_WINDOW", "");
                x = (wr.Left + wr.Right) / 2.0;
                y = (wr.Top + wr.Bottom) / 2.0;
            }
            Front(hwnd);
            Dictionary<string, object> o = D();
            o["hwnd"] = (long)Root(hwnd);
            o["x"] = (int)Math.Round(x);
            o["y"] = (int)Math.Round(y);
            return o;
        }

        private static Native.INPUT Mouse(uint flags, int dx, int dy, uint data)
        {
            Native.INPUT input = new Native.INPUT();
            input.type = 0;
            input.u.mi.dx = dx;
            input.u.mi.dy = dy;
            input.u.mi.mouseData = data;
            input.u.mi.dwFlags = flags;
            return input;
        }

        private static Native.INPUT KeyInput(ushort vk, ushort scan, uint flags)
        {
            Native.INPUT input = new Native.INPUT();
            input.type = 1;
            input.u.ki.wVk = vk;
            input.u.ki.wScan = scan;
            input.u.ki.dwFlags = flags;
            return input;
        }

        private static void Send(List<Native.INPUT> inputs)
        {
            if (inputs.Count == 0) return;
            Native.INPUT[] array = inputs.ToArray();
            uint sent = Native.SendInput((uint)array.Length, array, Marshal.SizeOf(typeof(Native.INPUT)));
            if (sent != array.Length) throw new LrError("INPUT_REFUSED", "");
        }

        /** MOUSEEVENTF_MOVE | ABSOLUTE | VIRTUALDESK: 0..65535 spans the whole virtual desktop, every monitor. */
        private static Native.INPUT MoveTo(int x, int y)
        {
            int vx = Native.GetSystemMetrics(76), vy = Native.GetSystemMetrics(77);
            int vw = Math.Max(2, Native.GetSystemMetrics(78)), vh = Math.Max(2, Native.GetSystemMetrics(79));
            int nx = (int)Math.Round((x - vx) * 65535.0 / (vw - 1));
            int ny = (int)Math.Round((y - vy) * 65535.0 / (vh - 1));
            return Mouse(0x0001 | 0x8000 | 0x4000, nx, ny, 0);
        }

        private static Dictionary<string, object> CursorJson()
        {
            Native.POINT p;
            if (!Native.GetCursorPos(out p))
            {
                NotSecure();
                throw new LrError("NO_POINTER", "");
            }
            Dictionary<string, object> o = D();
            o["x"] = p.X;
            o["y"] = p.Y;
            return o;
        }

        private static Dictionary<string, object> Click(Dictionary<string, object> a)
        {
            IntPtr hwnd = Hwnd(a);
            InputGuard(hwnd);
            int x = (int)Num(a, "x", 0), y = (int)Num(a, "y", 0);
            List<Native.INPUT> move = new List<Native.INPUT>();
            move.Add(MoveTo(x, y));
            Send(move);
            // Hover first, so menus and buttons that react to the pointer are ready for the press.
            Thread.Sleep(60);
            InputGuard(hwnd);
            List<Native.INPUT> press = new List<Native.INPUT>();
            press.Add(Mouse(0x0002, 0, 0, 0));
            press.Add(Mouse(0x0004, 0, 0, 0));
            Send(press);
            return CursorJson();
        }

        private static Dictionary<string, object> SelectAll(Dictionary<string, object> a)
        {
            InputGuard(Hwnd(a));
            List<Native.INPUT> keys = new List<Native.INPUT>();
            keys.Add(KeyInput(0x11, (ushort)Native.MapVirtualKey(0x11, 0), 0));
            keys.Add(KeyInput(0x41, (ushort)Native.MapVirtualKey(0x41, 0), 0));
            keys.Add(KeyInput(0x41, (ushort)Native.MapVirtualKey(0x41, 0), 0x0002));
            keys.Add(KeyInput(0x11, (ushort)Native.MapVirtualKey(0x11, 0), 0x0002));
            Send(keys);
            return D();
        }

        /** One chunk of text as KEYEVENTF_UNICODE events, a UTF-16 code unit each. Line breaks are never sent as keys. */
        private static Dictionary<string, object> TypeText(Dictionary<string, object> a)
        {
            string text = Str(a, "text") ?? "";
            if (text.IndexOf('\n') >= 0 || text.IndexOf('\r') >= 0) throw new LrError("NEWLINE", "");
            if (text.Length > 200) throw new LrError("TOO_LONG", "");
            InputGuard(Hwnd(a));
            List<Native.INPUT> keys = new List<Native.INPUT>();
            foreach (char c in text)
            {
                keys.Add(KeyInput(0, c, 0x0004));
                keys.Add(KeyInput(0, c, 0x0004 | 0x0002));
            }
            Send(keys);
            return D();
        }

        private static readonly int[] AllowedKeys = new int[] { 0x0D, 0x1B, 0x09, 0x20, 0x2E };

        private static Dictionary<string, object> Key(Dictionary<string, object> a)
        {
            int vk = (int)Num(a, "vk", 0);
            if (Array.IndexOf(AllowedKeys, vk) < 0) throw new LrError("UNKNOWN_OP", "key");
            InputGuard(Hwnd(a));
            ushort scan = (ushort)Native.MapVirtualKey((uint)vk, 0);
            List<Native.INPUT> keys = new List<Native.INPUT>();
            keys.Add(KeyInput((ushort)vk, scan, 0));
            keys.Add(KeyInput((ushort)vk, scan, 0x0002));
            Send(keys);
            return D();
        }

        private static Dictionary<string, object> Wheel(Dictionary<string, object> a)
        {
            IntPtr hwnd = Hwnd(a);
            InputGuard(hwnd);
            int notches = (int)Num(a, "notches", 0);
            if (notches == 0 || Math.Abs(notches) > 40) throw new LrError("UNKNOWN_OP", "wheel");
            List<Native.INPUT> move = new List<Native.INPUT>();
            move.Add(MoveTo((int)Num(a, "x", 0), (int)Num(a, "y", 0)));
            Send(move);
            Thread.Sleep(40);
            for (int n = 0; n < Math.Abs(notches); n++)
            {
                InputGuard(hwnd);
                List<Native.INPUT> wheel = new List<Native.INPUT>();
                wheel.Add(Mouse(0x0800, 0, 0, unchecked((uint)(notches > 0 ? 120 : -120))));
                Send(wheel);
                Thread.Sleep(15);
            }
            return CursorJson();
        }

        private static Dictionary<string, object> Pointer()
        {
            return CursorJson();
        }

        // ---- displays and screenshots ----

        private static List<object> Displays()
        {
            List<object> list = new List<object>();
            Native.EnumDisplayMonitors(IntPtr.Zero, IntPtr.Zero, delegate (IntPtr monitor, IntPtr hdc, ref Native.RECT clip, IntPtr data)
            {
                Native.MONITORINFO info = new Native.MONITORINFO();
                info.cbSize = Marshal.SizeOf(typeof(Native.MONITORINFO));
                if (Native.GetMonitorInfo(monitor, ref info))
                {
                    Dictionary<string, object> d = RectJson(info.rcMonitor.Left, info.rcMonitor.Top, info.rcMonitor.Right - info.rcMonitor.Left, info.rcMonitor.Bottom - info.rcMonitor.Top);
                    d["primary"] = (info.dwFlags & 1) != 0;
                    list.Add(d);
                }
                return true;
            }, IntPtr.Zero);
            return list;
        }

        /** The scale for a capture of w x h so its longer edge is at most maxEdge, never above 1 (fitScale in types.ts). */
        private static double Fit(int w, int h, int maxEdge)
        {
            int edge = Math.Max(w, h);
            return edge > maxEdge && edge > 0 ? (double)maxEdge / edge : 1.0;
        }

        private static Dictionary<string, object> Screenshot(Dictionary<string, object> a)
        {
            IntPtr hwnd = FrontWindow();
            uint pid = OwnerPid(hwnd);
            Native.RECT r = WindowRect(hwnd);
            int vx = Native.GetSystemMetrics(76), vy = Native.GetSystemMetrics(77);
            int vr = vx + Native.GetSystemMetrics(78), vb = vy + Native.GetSystemMetrics(79);
            int left = Math.Max(r.Left, vx), top = Math.Max(r.Top, vy), right = Math.Min(r.Right, vr), bottom = Math.Min(r.Bottom, vb);
            int w = right - left, h = bottom - top;
            if (w <= 0 || h <= 0) throw new LrError("NO_WINDOW", AppName(pid));
            int maxEdge = (int)Num(a, "maxEdge", 1568);
            double scale = Fit(w, h, maxEdge);
            int ow = Math.Max(1, (int)Math.Round(w * scale)), oh = Math.Max(1, (int)Math.Round(h * scale));
            string png;
            using (Bitmap full = new Bitmap(w, h, PixelFormat.Format32bppArgb))
            {
                using (Graphics g = Graphics.FromImage(full))
                {
                    g.CopyFromScreen(left, top, 0, 0, new Size(w, h), CopyPixelOperation.SourceCopy);
                }
                if (ow == w && oh == h) png = Png(full);
                else
                {
                    using (Bitmap small = new Bitmap(ow, oh, PixelFormat.Format32bppArgb))
                    {
                        using (Graphics g = Graphics.FromImage(small))
                        using (ImageAttributes edges = new ImageAttributes())
                        {
                            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                            g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                            g.CompositingQuality = CompositingQuality.HighQuality;
                            g.SmoothingMode = SmoothingMode.HighQuality;
                            // Mirrored edges keep the bicubic filter from darkening the border pixels.
                            edges.SetWrapMode(WrapMode.TileFlipXY);
                            g.DrawImage(full, new Rectangle(0, 0, ow, oh), 0, 0, w, h, GraphicsUnit.Pixel, edges);
                        }
                        png = Png(small);
                    }
                }
            }
            Dictionary<string, object> o = D();
            o["png"] = png;
            o["width"] = ow;
            o["height"] = oh;
            o["region"] = RectJson(left, top, w, h);
            o["app"] = AppName(pid);
            o["title"] = WindowText(hwnd);
            return o;
        }

        private static string Png(Bitmap bitmap)
        {
            using (MemoryStream stream = new MemoryStream())
            {
                bitmap.Save(stream, ImageFormat.Png);
                return Convert.ToBase64String(stream.ToArray());
            }
        }

        // ---- windows and apps ----

        /** Top-level windows a person can see and switch to, front to back (EnumWindows is in z-order). */
        private static List<object> Windows()
        {
            List<object> list = new List<object>();
            uint self = (uint)Process.GetCurrentProcess().Id;
            Dictionary<uint, string> names = new Dictionary<uint, string>();
            Native.EnumWindows(delegate (IntPtr hwnd, IntPtr unused)
            {
                if (!Native.IsWindowVisible(hwnd) || Native.GetWindow(hwnd, 4) != IntPtr.Zero) return true;
                if ((Native.GetWindowLong(hwnd, -20) & 0x80) != 0) return true;
                int cloaked;
                if (Native.DwmGetWindowAttributeInt(hwnd, 14, out cloaked, 4) == 0 && cloaked != 0) return true;
                string title = WindowText(hwnd);
                if (title.Length == 0) return true;
                uint pid = OwnerPid(hwnd);
                if (pid == self) return true;
                string app;
                if (!names.TryGetValue(pid, out app)) { app = AppName(pid); names[pid] = app; }
                Dictionary<string, object> w = D();
                w["hwnd"] = (long)hwnd;
                w["pid"] = (long)pid;
                w["app"] = app;
                w["process"] = ProcessName(pid);
                w["title"] = title;
                w["minimized"] = Native.IsIconic(hwnd);
                list.Add(w);
                return true;
            }, IntPtr.Zero);
            return list;
        }

        private static Dictionary<string, object> Foreground(Dictionary<string, object> a)
        {
            NotSecure();
            IntPtr hwnd = Hwnd(a);
            if (!Native.IsWindow(hwnd)) throw new LrError("GONE", "");
            Front(hwnd);
            return D();
        }

        /** WM_CLOSE, what CloseMainWindow sends: the app may ask about unsaved work. Never a kill. */
        private static Dictionary<string, object> Close(Dictionary<string, object> a)
        {
            IntPtr hwnd = Hwnd(a);
            if (!Native.IsWindow(hwnd)) return D();
            uint pid = OwnerPid(hwnd);
            if (!Native.PostMessage(hwnd, 0x0010, IntPtr.Zero, IntPtr.Zero) && Elevated(pid)) throw new LrError("ELEVATED", AppName(pid));
            return D();
        }

        // ---- browser tabs (Chrome, Edge) ----

        private static IntPtr BrowserWindow()
        {
            IntPtr front = Root(Native.GetForegroundWindow());
            IntPtr chosen = IntPtr.Zero;
            if (front != IntPtr.Zero && TabStripBrowser(OwnerPid(front))) return front;
            Native.EnumWindows(delegate (IntPtr hwnd, IntPtr unused)
            {
                if (!Native.IsWindowVisible(hwnd) || Native.GetWindow(hwnd, 4) != IntPtr.Zero || WindowText(hwnd).Length == 0) return true;
                if (!TabStripBrowser(OwnerPid(hwnd))) return true;
                chosen = hwnd;
                return false;
            }, IntPtr.Zero);
            return chosen;
        }

        private static bool TabStripBrowser(uint pid)
        {
            string name = ProcessName(pid);
            return string.Equals(name, "chrome", StringComparison.OrdinalIgnoreCase) || string.Equals(name, "msedge", StringComparison.OrdinalIgnoreCase);
        }

        /** The tab strip's tabs: TabItems inside a Tab control, never inside the page (a Document). */
        private static List<AutomationElement> TabItems(IntPtr hwnd)
        {
            List<AutomationElement> tabs = new List<AutomationElement>();
            Queue<AutomationElement> queue = new Queue<AutomationElement>();
            queue.Enqueue(AutomationElement.FromHandle(hwnd));
            int visited = 0;
            Stopwatch clock = Stopwatch.StartNew();
            while (queue.Count > 0 && visited < 1500 && clock.ElapsedMilliseconds < 3000)
            {
                AutomationElement node = queue.Dequeue();
                AutomationElementCollection kids;
                try { kids = node.FindAll(TreeScope.Children, Automation.ControlViewCondition); }
                catch (ElementNotAvailableException) { continue; }
                foreach (AutomationElement el in kids)
                {
                    visited++;
                    ControlType type;
                    try { type = el.Current.ControlType; }
                    catch (ElementNotAvailableException) { continue; }
                    if (type == ControlType.Document) continue;
                    if (type == ControlType.TabItem) { tabs.Add(el); continue; }
                    queue.Enqueue(el);
                }
            }
            return tabs;
        }

        private static Dictionary<string, object> Tabs()
        {
            NotSecure();
            IntPtr hwnd = BrowserWindow();
            Dictionary<string, object> o = D();
            List<object> list = new List<object>();
            o["tabs"] = list;
            if (hwnd == IntPtr.Zero) { o["app"] = ""; return o; }
            uint pid = OwnerPid(hwnd);
            o["app"] = AppName(pid);
            if (Elevated(pid)) throw new LrError("ELEVATED", AppName(pid));
            List<AutomationElement> tabs = TabItems(hwnd);
            for (int n = 0; n < tabs.Count; n++)
            {
                Dictionary<string, object> t = D();
                t["index"] = n;
                try
                {
                    t["title"] = tabs[n].Current.Name ?? "";
                    object p;
                    t["active"] = tabs[n].TryGetCurrentPattern(SelectionItemPattern.Pattern, out p) && ((SelectionItemPattern)p).Current.IsSelected;
                }
                catch (ElementNotAvailableException) { continue; }
                list.Add(t);
            }
            return o;
        }

        /** Selects a tab through SelectionItem (or Invoke) without the pointer, then brings the browser forward. */
        private static Dictionary<string, object> TabSelect(Dictionary<string, object> a)
        {
            NotSecure();
            IntPtr hwnd = BrowserWindow();
            if (hwnd == IntPtr.Zero) throw new LrError("NO_TAB", "");
            NotElevated(hwnd);
            List<AutomationElement> tabs = TabItems(hwnd);
            int index = (int)Num(a, "index", -1);
            if (index < 0 || index >= tabs.Count || (tabs[index].Current.Name ?? "") != (Str(a, "title") ?? "")) throw new LrError("NO_TAB", "");
            object p;
            AutomationElement tab = tabs[index];
            if (tab.TryGetCurrentPattern(SelectionItemPattern.Pattern, out p)) ((SelectionItemPattern)p).Select();
            else if (tab.TryGetCurrentPattern(InvokePattern.Pattern, out p)) ((InvokePattern)p).Invoke();
            else throw new LrError("NO_TAB", "");
            Front(hwnd);
            return D();
        }

        // ---- volume (Core Audio) ----

        private static object Activate(IMMDevice device, Guid iid)
        {
            object instance;
            Marshal.ThrowExceptionForHR(device.Activate(ref iid, 23, IntPtr.Zero, out instance));
            return instance;
        }

        private static IMMDevice Speakers()
        {
            IMMDeviceEnumerator devices = (IMMDeviceEnumerator)new MMDeviceEnumeratorCom();
            IMMDevice device;
            if (devices.GetDefaultAudioEndpoint(0, 1, out device) != 0 || device == null) throw new LrError("NO_AUDIO", "");
            return device;
        }

        private static ISimpleAudioVolume Session(string app)
        {
            IAudioSessionManager2 manager = (IAudioSessionManager2)Activate(Speakers(), new Guid("77AA99A0-1BD6-484F-8BC7-2C654C9A9B6F"));
            IAudioSessionEnumerator sessions;
            Marshal.ThrowExceptionForHR(manager.GetSessionEnumerator(out sessions));
            int count;
            Marshal.ThrowExceptionForHR(sessions.GetCount(out count));
            for (int n = 0; n < count; n++)
            {
                object session;
                if (sessions.GetSession(n, out session) != 0 || session == null) continue;
                IAudioSessionControl2 control = session as IAudioSessionControl2;
                uint pid;
                if (control == null || control.GetProcessId(out pid) != 0 || pid == 0) continue;
                if (string.Equals(ProcessName(pid), app, StringComparison.OrdinalIgnoreCase)) return (ISimpleAudioVolume)session;
            }
            throw new LrError("NOT_PLAYING", app);
        }

        private static Dictionary<string, object> Level(float scalar, bool muted)
        {
            Dictionary<string, object> o = D();
            o["level"] = (int)Math.Round(scalar * 100);
            o["muted"] = muted;
            return o;
        }

        private static Dictionary<string, object> VolumeRead(Dictionary<string, object> a)
        {
            float level;
            bool muted;
            if (Str(a, "target") == "Spotify")
            {
                ISimpleAudioVolume session = Session("Spotify");
                Marshal.ThrowExceptionForHR(session.GetMasterVolume(out level));
                Marshal.ThrowExceptionForHR(session.GetMute(out muted));
                return Level(level, muted);
            }
            IAudioEndpointVolume volume = (IAudioEndpointVolume)Activate(Speakers(), new Guid("5CDF2C82-841E-4546-9722-0CF74078229A"));
            Marshal.ThrowExceptionForHR(volume.GetMasterVolumeLevelScalar(out level));
            Marshal.ThrowExceptionForHR(volume.GetMute(out muted));
            return Level(level, muted);
        }

        private static Dictionary<string, object> VolumeWrite(Dictionary<string, object> a)
        {
            bool hasLevel = Has(a, "level"), hasMuted = Has(a, "muted");
            float level = (float)(Math.Max(0, Math.Min(100, Num(a, "level", 0))) / 100.0);
            object mutedValue;
            bool muted = a.TryGetValue("muted", out mutedValue) && mutedValue is bool && (bool)mutedValue;
            if (Str(a, "target") == "Spotify")
            {
                ISimpleAudioVolume session = Session("Spotify");
                if (hasLevel) Marshal.ThrowExceptionForHR(session.SetMasterVolume(level, IntPtr.Zero));
                if (hasMuted) Marshal.ThrowExceptionForHR(session.SetMute(muted, IntPtr.Zero));
            }
            else
            {
                IAudioEndpointVolume volume = (IAudioEndpointVolume)Activate(Speakers(), new Guid("5CDF2C82-841E-4546-9722-0CF74078229A"));
                if (hasLevel) Marshal.ThrowExceptionForHR(volume.SetMasterVolumeLevelScalar(level, IntPtr.Zero));
                if (hasMuted) Marshal.ThrowExceptionForHR(volume.SetMute(muted, IntPtr.Zero));
            }
            return VolumeRead(a);
        }

        // ---- dark mode, lock, media keys ----

        private const string Personalize = "Software\\Microsoft\\Windows\\CurrentVersion\\Themes\\Personalize";

        private static Dictionary<string, object> DarkMode(Dictionary<string, object> a)
        {
            string action = Str(a, "action");
            bool dark;
            using (RegistryKey key = Registry.CurrentUser.CreateSubKey(Personalize))
            {
                object light = key.GetValue("AppsUseLightTheme");
                bool isDark = light is int && (int)light == 0;
                if (action == "on") dark = true;
                else if (action == "off") dark = false;
                else if (action == "toggle") dark = !isDark;
                else throw new LrError("UNKNOWN_OP", action);
                key.SetValue("AppsUseLightTheme", dark ? 0 : 1, RegistryValueKind.DWord);
                key.SetValue("SystemUsesLightTheme", dark ? 0 : 1, RegistryValueKind.DWord);
            }
            // Open apps re-read the theme when told the "ImmersiveColorSet" setting changed.
            IntPtr result;
            Native.SendMessageTimeout(new IntPtr(0xffff), 0x001A, IntPtr.Zero, "ImmersiveColorSet", 0x0002, 3000, out result);
            Dictionary<string, object> o = D();
            o["dark"] = dark;
            return o;
        }

        private static Dictionary<string, object> Lock()
        {
            if (!Native.LockWorkStation()) throw new LrError("LOCK_FAILED", "");
            return D();
        }

        private static Dictionary<string, object> MediaKey(Dictionary<string, object> a)
        {
            string action = Str(a, "action");
            ushort vk = action == "next" ? (ushort)0xB0 : action == "previous" ? (ushort)0xB1 : (ushort)0xB3;
            NotSecure();
            List<Native.INPUT> keys = new List<Native.INPUT>();
            keys.Add(KeyInput(vk, 0, 0x0001));
            keys.Add(KeyInput(vk, 0, 0x0001 | 0x0002));
            Send(keys);
            return D();
        }

        // ---- access ----

        private static Dictionary<string, object> Access()
        {
            Dictionary<string, object> o = D();
            bool uia = false;
            try
            {
                AutomationElement root = AutomationElement.RootElement;
                uia = root != null && root.Current.ControlType == ControlType.Pane;
            }
            catch (ElementNotAvailableException) { }
            catch (COMException) { }
            catch (InvalidOperationException) { }
            uint session;
            Native.ProcessIdToSessionId((uint)Process.GetCurrentProcess().Id, out session);
            o["uia"] = uia;
            o["secure"] = SecureDesktop();
            o["session"] = (long)session;
            o["integrity"] = OwnIntegrity();
            o["dpi"] = dpi;
            o["interactive"] = Environment.UserInteractive;
            return o;
        }
    }
}
`
