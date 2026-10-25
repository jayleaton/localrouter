import { HELPER_CS } from './helper.cs.ts'

/** Assemblies the C# helper is compiled against (Windows PowerShell 5.1 resolves these names from the GAC). */
export const REFERENCED_ASSEMBLIES = ['UIAutomationClient', 'UIAutomationTypes', 'System.Drawing', 'System.Windows.Forms', 'WindowsBase']

/**
 * Operations that run in PowerShell itself rather than in the C# helper: the ones that need a cmdlet (Get-StartApps)
 * or WinRT, which Windows PowerShell 5.1 projects for free and .NET Framework C# compiled by Add-Type cannot reach
 * without the Windows metadata files. Each is a script block taking the request's args (a parsed dictionary): no
 * request value is ever part of the script text. Each returns one value (a hashtable or a list) for the reply.
 */
export const PS_OPS: Record<string, string> = {
  start_apps: String.raw`param($a)
$list = New-Object System.Collections.ArrayList
foreach ($app in @(Get-StartApps)) { $null = $list.Add(@{ name = [string]$app.Name; id = [string]$app.AppID }) }
, $list`,

  media: String.raw`param($a)
$null = [Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager, Windows.Media.Control, ContentType = WindowsRuntime]
$manager = Wait-LrAsync ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]::RequestAsync()) ([Windows.Media.Control.GlobalSystemMediaTransportControlsSessionManager]) 10000
$session = $null
if ($a['player'] -eq 'Spotify') {
  foreach ($candidate in $manager.GetSessions()) { if ($candidate.SourceAppUserModelId -like '*Spotify*') { $session = $candidate; break } }
} else {
  $session = $manager.GetCurrentSession()
}
if ($null -eq $session) { return @{ session = $false; done = $false; app = '' } }
switch ([string]$a['action']) {
  'play' { $operation = $session.TryPlayAsync() }
  'pause' { $operation = $session.TryPauseAsync() }
  'next' { $operation = $session.TrySkipNextAsync() }
  'previous' { $operation = $session.TrySkipPreviousAsync() }
  default { throw 'LR_UNKNOWN_OP|media|' }
}
$done = Wait-LrAsync $operation ([bool]) 10000
@{ session = $true; done = [bool]$done; app = [string]$session.SourceAppUserModelId }`,

  hello_available: String.raw`param($a)
$null = [Windows.Security.Credentials.UI.UserConsentVerifier, Windows.Security.Credentials.UI, ContentType = WindowsRuntime]
$availability = Wait-LrAsync ([Windows.Security.Credentials.UI.UserConsentVerifier]::CheckAvailabilityAsync()) ([Windows.Security.Credentials.UI.UserConsentVerifierAvailability]) 10000
@{ availability = [string]$availability }`,

  hello_verify: String.raw`param($a)
$null = [Windows.Security.Credentials.UI.UserConsentVerifier, Windows.Security.Credentials.UI, ContentType = WindowsRuntime]
$result = Wait-LrAsync ([Windows.Security.Credentials.UI.UserConsentVerifier]::RequestVerificationAsync([string]$a['prompt'])) ([Windows.Security.Credentials.UI.UserConsentVerificationResult]) 120000
@{ result = [string]$result }`,
}

const base64 = (text: string) => Buffer.from(text, 'utf8').toString('base64')

/**
 * The worker script. It compiles the helper, makes the process DPI aware (Lr.Hello, before any UI Automation or GDI
 * call), prints a ready reply (id 0), then answers one JSON request a line until stdin closes. Nothing but replies is
 * written to stdout: every other value is assigned to $null. Errors become `ok: false` replies with the innermost
 * exception's message, which carries the helper's `LR_CODE|detail|` when it is one of ours.
 */
export function bootstrapScript(source: string = HELPER_CS, assemblies: string[] = REFERENCED_ASSEMBLIES): string {
  const ops = Object.entries(PS_OPS)
    .map(([name, body]) => `$ops['${name}'] = [scriptblock]::Create([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${base64(body)}')))`)
    .join('\n')
  return String.raw`$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$WarningPreference = 'SilentlyContinue'
try {
  $source = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${base64(source)}'))
  Add-Type -TypeDefinition $source -Language CSharp -ReferencedAssemblies @(${assemblies.map((name) => `'${name}'`).join(', ')})
  $hello = [LocalRouterWin.Lr]::Hello()
} catch {
  $message = $_.Exception.GetBaseException().Message
  [Console]::Out.WriteLine('{"id":0,"ok":false,"error":' + (ConvertTo-Json -InputObject ([string]$message) -Compress) + '}')
  [Console]::Out.Flush()
  exit 1
}
$LrAsTask = $null
function Wait-LrAsync($operation, [Type]$type, [int]$ms) {
  if ($null -eq $script:LrAsTask) {
    $null = Add-Type -AssemblyName System.Runtime.WindowsRuntime
    $generic = 'IAsyncOperation' + [char]96 + '1'
    $script:LrAsTask = [System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq $generic } | Select-Object -First 1
  }
  $task = $script:LrAsTask.MakeGenericMethod($type).Invoke($null, @($operation))
  if (-not $task.Wait($ms)) { throw 'LR_TIMEOUT||' }
  $task.Result
}
$ops = @{}
${ops}
[Console]::Out.WriteLine('{"id":0,"ok":true,"result":' + $hello + '}')
[Console]::Out.Flush()
while ($true) {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { break }
  if ($line.Trim().Length -eq 0) { continue }
  $id = -1
  try {
    $request = [LocalRouterWin.Lr]::Parse($line)
    $id = $request.Id
    if ($ops.ContainsKey($request.Op)) {
      $value = & $ops[$request.Op] $request.Args
      $result = [LocalRouterWin.Json]::Write($value)
    } else {
      $result = [LocalRouterWin.Lr]::Run($request.Op, $request.Args)
    }
    $reply = '{"id":' + $id + ',"ok":true,"result":' + $result + '}'
  } catch {
    $reply = '{"id":' + $id + ',"ok":false,"error":' + [LocalRouterWin.Json]::Quote($_.Exception.GetBaseException().Message) + '}'
  }
  [Console]::Out.WriteLine($reply)
  [Console]::Out.Flush()
}
`
}

/**
 * The one line sent on stdin first. `powershell.exe -Command -` runs stdin a line at a time, and a multi-line block
 * would need a blank line to end it, so the whole script travels as one base64 line and runs with Invoke-Expression;
 * its loop then reads the following lines (the requests) from the same stdin.
 */
export function bootstrapLine(script: string = bootstrapScript()): string {
  return `Invoke-Expression ([Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('${base64(script)}')))`
}
