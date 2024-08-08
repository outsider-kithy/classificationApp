# React Native(Expo) + TensorFlow.jsで物体検出アプリ
Coco-SSDを利用して、画像から物体を検出するスマホアプリ

## デモ画面
![デモ画面](demo_img.PNG)

## バージョン
- "@tensorflow-models/coco-ssd": "^2.2.3",
- "@tensorflow/tfjs": "^4.20.0",
- "@tensorflow/tfjs-backend-cpu": "^4.20.0",
- "@tensorflow/tfjs-backend-webgl": "^4.20.0",
- "@tensorflow/tfjs-react-native": "^1.0.0",
- "expo": "~51.0.24",
- "expo-status-bar": "~1.12.1",
- "react": "18.2.0",
- "react-native": "0.74.3",
- "react-native-canvas": "^0.1.40",
- "react-native-webview": "13.8.6"

## 注意点(2024年8月時点)

`tfjs-react-native`のCameraコンポーネントには最新の`react-native`との間に互換性がないので、`@tensorflow/tfjs-react-native/dist/index.js`内でコメントアウトする必要あり。

```JavaScript:@tensorflow/tfjs-react-native/dist/index.js
/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
import './platform_react_native';
export { asyncStorageIO } from './async_storage_io';
export { bundleResourceIO } from './bundle_resource_io';
export { decodeJpeg } from './decode_image';
export { fetch } from './platform_react_native';
export { version } from './version';
// 以下をコメントアウト！
// export * from './camera/camera';
// export * from './camera/camera_stream';
//# sourceMappingURL=index.js.map
```