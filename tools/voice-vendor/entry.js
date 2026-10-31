// The /voice page's 3D viewer: the few three.js and Spark exports it uses, bundled into one ESM file by build.mjs.
export {
  WebGLRenderer,
  Scene,
  PerspectiveCamera,
  OrthographicCamera,
  Vector3,
  Color,
} from 'three';
export { OrbitControls } from 'three/addons/controls/OrbitControls.js';
export { SparkRenderer, SplatMesh, SplatFileType } from '@sparkjsdev/spark';
export const versions = { three: '0.187.0', spark: '2.4.0' };
