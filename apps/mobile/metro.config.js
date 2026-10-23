// Expo's default Metro config, with zod's locales trimmed to English.
const path = require('path')
const { getDefaultConfig } = require('expo/metro-config')

const config = getDefaultConfig(__dirname)

// `import { z } from 'zod'` re-exports every locale, and tree shaking cannot drop them. The app never switches
// locale, so zod's own imports of its locales index get English only (about 270 KB of minified JS).
const zodLocales = path.join(__dirname, 'metro/zod-locales.js')
const insideZod = /[\\/]zod[\\/]v4[\\/](classic|core|mini)[\\/]/
const resolveRequest = config.resolver.resolveRequest
config.resolver.resolveRequest = (context, moduleName, platform) => {
  if (/^\.\.\/locales\/index\.c?js$/.test(moduleName) && insideZod.test(context.originModulePath)) {
    return { type: 'sourceFile', filePath: zodLocales }
  }
  return (resolveRequest ?? context.resolveRequest)(context, moduleName, platform)
}

module.exports = config
