import React, { useEffect, useRef, useState } from 'react';
import { StyleSheet, Text, Image, View, SafeAreaView } from 'react-native';
import * as tf from "@tensorflow/tfjs";
import { fetch, decodeJpeg } from '@tensorflow/tfjs-react-native';
// Note: Require the cpu and webgl backend and add them to package.json as peer dependencies.
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';
import * as cocoSsd from "@tensorflow-models/coco-ssd";

import Canvas from 'react-native-canvas';

export default function App() {
	const [isTfReady, setIsTfReady] = useState(false);
	const [result, setResult] = useState("");
	const image = useRef(null);
	const canvasRef = useRef(null);

	const [imgWidth, imgHeight]	= [300, 300];
	const imagePath = "./assets/judith-prins-AJa7S1fjy-I-unsplash.jpg";

	//物体のエリアに四角形を描画する関数
	const drawRect = (predictions, ctx) =>{
		//検出され物体の数だけ繰り返す
		predictions.forEach(prediction => {

			// バウンディングボックスの座標とサイズを取得
			const [x, y, width, height] = prediction['bbox'];
			const text = prediction['class']; 
			//console.log(prediction);

			// Canvasの設定
			ctx.strokeStyle = '#00ff00';
			ctx.lineWidth = "2";

			//　物体の名前を描画
			ctx.fillStyle = '#00ff00';
			ctx.font = '18px Arial';
			ctx.fillText(text, x, y);

			// 四角形を描画
			ctx.beginPath();
			ctx.strokeRect(x, y, width, height);
			ctx.closePath();
		});
	}

	const load = async () => {
		try{
			//モデルの読み込み
			await tf.ready();
			const model = await cocoSsd.load();
			setIsTfReady(true);
			
			//画像の読み込み・テンソル化
			const image = require(imagePath);
			const imageAssetPath = Image.resolveAssetSource(image);
			const response = await fetch(imageAssetPath.uri, {}, {isBinary: true});
			const imageDataArrayBuffer = await response.arrayBuffer();
			const imageData = new Uint8Array(imageDataArrayBuffer);
			const imageTensor = decodeJpeg(imageData);
			//imageTensor.print();
		
			//画像のリサイズ
			const resizedTensor = tf.image.resizeBilinear(imageTensor, [imgWidth, imgHeight]);
			//resizedTensor.print();
			const castedTensor = resizedTensor.toInt();

			//物体検出
			const predictions = await model.detect(castedTensor);

			//少なくとも1つの物体が検出されたら
			if(predictions && predictions.length > 0){

				//predictionの中身はオブジェクトの配列
				let predictionArray = [];

				// 検出された物体の名前を配列に格納
				for(let i = 0; i < predictions.length; i++){
					//console.log(predictions[i]);
					predictionArray.push(predictions[i].class);
				}
				setResult(`${predictionArray.join(", ")}`);

				// Canvasの大きさを画像の大きさに合わせる
				canvasRef.current.width = imgWidth;
				canvasRef.current.height = imgHeight;
		
				// 検出結果を描画
				const ctx = canvasRef.current.getContext("2d");
				drawRect(predictions, ctx); 
			}

		//テンソルの解放
		imageTensor.dispose();
		resizedTensor.dispose();
		castedTensor.dispose();
		model.dispose();

		} catch(err){
			console.log(err);
		}
	}

	useEffect(() => {
		load();
	}, []);


  return (
	<SafeAreaView style={styles.container}>
		<Text style={styles.h1}>画像から物体検出</Text>
		<View style={styles.box}>
			<Image
				ref={image}
				source={require(imagePath)}
				style={styles.image}
			/>
			<Canvas 
				ref={canvasRef} 
				style={styles.canvas} 
			/>
		</View>
		{!isTfReady && <Text>モデルを読み込み中...</Text>}
		{!isTfReady || result === "" && <Text>物体を検出中...</Text>}
		{result !== "" && <Text style={styles.result}>検出結果：{result}</Text>}
	</SafeAreaView>
  );
}

const styles = StyleSheet.create({
	container: {
		flex: 1,
		alignItems: 'center',
		justifyContent: 'center',
	  },
	  h1:{
		fontSize: 24,
		fontWeight: 'bold',
		marginBottom: 20,
	  },
	  box:{
		position: 'relative',
		width: 300,
		height: 300,
		marginBottom: 20,
	  },
	  image:{
		position: 'absolute',
		top: 0,
		width: 300,
		height: 300,
		zIndex: 0,
	  },
	  canvas:{
		position: 'absolute',
		top: 0,
		width: 300,
		height: 300,
		zIndex: 1,
	  },
	  result: {
		marginTop: 20,
		fontWeight: 'bold',
		fontSize: 20,
		color: 'red',
	  }
});
