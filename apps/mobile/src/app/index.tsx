import { Redirect } from 'expo-router'

import { useBox } from '@/lib/app/box'

/** Entry route: home when a box is paired, pairing otherwise. */
export default function Index() {
  const status = useBox((box) => box.status)
  return <Redirect href={status === 'paired' ? '/home' : '/pair'} />
}
