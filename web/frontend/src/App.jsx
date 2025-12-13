import React, { useRef, useState, useEffect } from "react";
import './App.css';

function App() {
  const [result, setResult] = useState(null);
  return (
    <>
      <h1 style={{ fontSize: "2.5em" }}>한자 필기 인식기</h1>
      <p>데이터셋 출처: <a href="https://nlpr.ia.ac.cn/databases/handwriting/home.html">CASIA Online and Offline Chinese Handwriting Databases</a></p>
      <div id="container">
        <Canvas setResult={setResult}/>
        <PredictList data={result}/>
      </div>
    </>
  );
}

function Canvas({setResult}) {
  const canvasRef = useRef(null);
  const ctxRef = useRef(null);
  const drawing = useRef(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    ctx.lineCap = "round";
    ctx.lineWidth = 5;
    ctx.strokeStyle = "black";
    ctxRef.current = ctx;
  }, []);

  const startDrawing = (e) => {
    drawing.current = true;
    draw(e);
  };

  const endDrawing = () => {
    drawing.current = false;
    ctxRef.current.beginPath();
  };

  const draw = (e) => {
    if (!drawing.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctxRef.current.lineTo(x, y);
    ctxRef.current.stroke();
    ctxRef.current.beginPath();
    ctxRef.current.moveTo(x, y);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = ctxRef.current;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    setResult(null);
  };

  const predict = async () => {
    const canvas = canvasRef.current;

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = 64;
    tempCanvas.height = 64;
    const tempCtx = tempCanvas.getContext("2d");
    tempCtx.fillStyle = "white";
    tempCtx.fillRect(0, 0, 64, 64);
    tempCtx.drawImage(canvas, 0, 0, 64, 64);

    const dataURL = tempCanvas.toDataURL("image/png");

    const res = await fetch("https://0.0.0.0:10000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataURL })
    });

    const json = await res.json();
    setResult(json);
  };

  return (
    <div style={{ marginTop: "3em" }}>
      <canvas
        ref={canvasRef}
        width={256}
        height={256}
        style={{ border: "1px solid black", background: "white" }}
        onMouseDown={startDrawing}
        onMouseUp={endDrawing}
        onMouseMove={draw}
        onMouseLeave={endDrawing}
      />

      <br/><br/>

      <button onClick={clearCanvas}>Clear</button>
      <button onClick={predict} style={{ marginLeft: "10px" }}>Predict</button>
    </div>
  );
}

function PredictList({data}) {
  return (
    <div style={{ width: "20em" }}>
      <ol style={{ display: "inline-block" }}>
        <legend >
          <h2>Predicted</h2>
        </legend>
        {data && data.topk.map((item, index) => (
          <li key={index} style={{ listStyle: "none", marginTop: "0.5em" }}>
            <Card item={item}/>
          </li>
        ))}
      </ol>
    </div>
  );
}

function Card({item}) {
  return (
    <table className="hanja-info">
      <tbody>
        <tr>
          <th className="hanja" rowSpan={2}>
            {item.hanja}
          </th>
          <th>Class</th>
          <td>{item.class}</td>
        </tr>
        <tr>
          <th>Probability</th>
          <td>{(item.probability * 100).toFixed(2)}%</td>
        </tr>
      </tbody>
    </table>
  );
}

export default App;
