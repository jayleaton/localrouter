import { contextBridge, ipcRenderer } from 'electron'

/**
 * What this app's pages may ask of it: nothing else crosses. The voice page gets the realtime bridge (its socket to the
 * box goes through the harness, in client tool mode); the setup page gets pairing and settings. The main process
 * checks which page asks before answering, and the device token never reaches either.
 */
type Listener = { onOpen?: () => void; onMessage?: (text: string) => void; onClose?: (code: number, reason: string) => void }
const listeners = new Map<string, Listener>()

ipcRenderer.on('rt:open', (_e, id: string) => listeners.get(id)?.onOpen?.())
ipcRenderer.on('rt:event', (_e, id: string, text: string) => listeners.get(id)?.onMessage?.(text))
ipcRenderer.on('rt:close', (_e, id: string, code: number, reason: string) => {
  const l = listeners.get(id)
  listeners.delete(id)
  l?.onClose?.(code, reason)
})

contextBridge.exposeInMainWorld('companion', {
  realtime: {
    async open(listener: Listener): Promise<string> {
      const id = (await ipcRenderer.invoke('rt:open')) as string
      listeners.set(id, listener)
      return id
    },
    send: (id: string, text: string) => ipcRenderer.send('rt:send', id, text),
    close: (id: string) => ipcRenderer.send('rt:close', id),
  },
  // The page's host: this computer's tools for typed chat, the journal, and a video moved from a call.
  host: {
    tools: () => ipcRenderer.invoke('host:tools'),
    run: (callId: string, name: string, args: string, utterance: string) => ipcRenderer.invoke('host:run', callId, name, args, utterance),
    cancel: (callId: string) => ipcRenderer.invoke('host:cancel', callId),
    save: (conversation: string, items: unknown[]) => ipcRenderer.invoke('host:save', conversation, items),
    prepareVideo: () => ipcRenderer.invoke('host:prepareVideo'),
    restoreVoice: () => ipcRenderer.invoke('host:restoreVoice'),
  },
  setup: {
    state: () => ipcRenderer.invoke('setup:state'),
    checkAddress: (address: string) => ipcRenderer.invoke('setup:checkAddress', address),
    pair: (code: string, name: string) => ipcRenderer.invoke('setup:pair', code, name),
    check: () => ipcRenderer.invoke('setup:check'),
    repair: () => ipcRenderer.invoke('setup:repair'),
    cancel: () => ipcRenderer.invoke('setup:cancel'),
    disconnect: () => ipcRenderer.invoke('setup:disconnect'),
    discover: () => ipcRenderer.invoke('setup:discover'),
    permission: (kind: string) => ipcRenderer.invoke('setup:permission', kind),
    resetGrant: (kind: string) => ipcRenderer.invoke('setup:resetGrant', kind),
    relaunch: () => ipcRenderer.invoke('setup:relaunch'),
    addFolder: () => ipcRenderer.invoke('setup:addFolder'),
    removeFolder: (path: string) => ipcRenderer.invoke('setup:removeFolder', path),
    enroll: (name: string) => ipcRenderer.invoke('setup:enroll', name),
    revoke: (id: string) => ipcRenderer.invoke('setup:revoke', id),
    legacy: (enabled: boolean, code: string) => ipcRenderer.invoke('setup:legacy', enabled, code),
    control: (on: boolean) => ipcRenderer.invoke('setup:control', on),
    exportConversations: () => ipcRenderer.invoke('setup:export'),
    openVoice: () => ipcRenderer.invoke('setup:openVoice'),
  },
})
