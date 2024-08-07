import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, Image, View } from 'react-native';
import * as tf from "@tensorflow/tfjs";
import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export default function App() {
  const [isTfReady, setIsTfReady] = useState(false);
  const [result, setResult] = useState("");
  const image = useRef(null);

  const load = async () => {
    try{
      //モデルの読み込み
      await tf.ready();
      const model = await cocoSsd.load();
      setIsTfReady(true);

      //画像の読み込み・テンソル化
      const image = require("./assets/myImage02.jpg");
      const imageAssetPath = Image.resolveAssetSource(image);
      const response = await fetch(imageAssetPath.uri, {}, {isBinary: true});
      const imageDataArrayBuffer = await response.arrayBuffer();
      const imageData = new Uint8Array(imageDataArrayBuffer);
      const imageTensor = decodeJpeg(imageData);

      const prediction = await model.detect(imageTensor);
      //少なくとも1つの物体が検出されたら
      if(prediction && prediction.length > 0){
        //predictionの中身はオブジェクトの配列
        let predictionArray = [];
        for(let i = 0; i < prediction.length; i++){
          console.log(prediction[i]);
          predictionArray.push(prediction[i].class);
        }

        setResult(
          `${predictionArray.join(", ")}`
        );
        //console.log(prediction);
      }

      //テンソルの解放
      imageTensor.dispose();
      model.dispose();

    } catch(err){
      console.log(err);
    }
  }

  useEffect(() => {
    load();
  }, []);


  return (
    <View style={styles.container}>
      <Image
        ref={image}
        source={require("./assets/myImage02.jpg")}
        style={styles.image}
      />

        {!isTfReady && <Text>モデルを読み込み中...</Text>}
        {!isTfReady || result === "" && <Text>物体を検出中...</Text>}
        {result !== "" && <Text style={styles.result}>{result}</Text>}
      
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    justifyContent: 'center',
  },
  image:{
    width: 300,
    height: 300,
  },
  result: {
    marginTop: 20,
    fontWeight: 'bold',
    fontSize: 20,
    color: 'red',
  }
});
