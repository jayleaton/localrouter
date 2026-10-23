import { useMemo } from 'react'

import { useBox } from '@/lib/app/box'

/**
 * A box file (a picture or video under /v1/files) as an image or video source. The pairing token goes along, so
 * the files still load once the box requires pairing for them.
 */
export function useFileSource(uri: string | undefined) {
  const token = useBox((box) => box.pairing?.token ?? null)
  return useMemo(() => (uri ? { uri, ...(token ? { headers: { authorization: `Bearer ${token}` } } : {}) } : null), [uri, token])
}
