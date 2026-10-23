import type { ConfigContext, ExpoConfig } from 'expo/config'

/*
 * APP_VARIANT=development gives the development build its own name and identifier, so it installs beside a
 * release build. EAS_PROJECT_ID links builds to the owner's own Expo project (`eas init` prints it); it is not
 * kept in the repository.
 */
export default ({ config }: ConfigContext): ExpoConfig => {
  const variant = process.env.APP_VARIANT ?? 'production'
  if (!['production', 'development'].includes(variant)) throw new Error(`Unknown APP_VARIANT: ${variant}`)
  const projectId = process.env.EAS_PROJECT_ID
  const extra = { ...config.extra, ...(projectId ? { eas: { projectId } } : {}) }
  if (variant === 'production') return { ...config, extra } as ExpoConfig
  const id = 'app.localrouter.mobile.dev'
  return {
    ...config,
    name: 'LocalRouter Dev',
    slug: config.slug ?? 'localrouter-mobile',
    android: { ...config.android, package: id },
    ios: { ...config.ios, bundleIdentifier: id },
    extra,
  }
}
