import React, { useEffect, useState } from 'react';
import pima_data from './data/diabetes.json';
const tf = require('@tensorflow/tfjs');

function App() {

  const [model, setModel] = useState(null);
  // const [ys, setYs] = useState(null);

  const [prag, setPrag] = useState('');
  const [glu, setGlu] = useState('');
  const [blo, setBlo] = useState('');
  const [ski, setSki] = useState('');
  const [ins, setIns] = useState('');
  const [bmi, setBmi] = useState('');
  const [dia, setDia] = useState('');
  const [age, setAge] = useState('');

  const [output, setOutput] = useState('');


  const main_xs = [];
  const main_ys = [];



  pima_data.map((data) => {

    let col = [data.Pregnancies, data.Glucose, data.BloodPressure,
    data.SkinThickness, data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age];

    main_xs.push(col);
    main_ys.push(data.Outcome);

  });

  // console.log(main_xs.length);

  // setXs(main_xs);
  // setYs(main_ys);

  // const aa = [[1, 2], [2, 3]];
  const xs_ = tf.tensor2d(main_xs, [768, 8]);
  const ys_ = tf.tensor1d(main_ys);
  // console.log(main_xs);
  // const ys_ = tf.constraints(main_ys);

  const train = async () => {

    // Train a simple model:
    const model = tf.sequential();
    model.add(tf.layers.dense({ units: 100, activation: 'relu', inputShape: [8] }));
    model.add(tf.layers.dense({ units: 1, activation: 'linear' }));
    model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });


    const xs = tf.randomNormal([100, 10]);
    const ys = tf.randomNormal([100, 1]);

    let epo = 0;


    await model.fit(xs_, ys_, {
      epochs: 100,
      callbacks: {
        onEpochEnd: (epoch, log) => { console.log(`Epoch ${epoch}: loss = ${log.loss}`); epo = epoch; }
      }
    });

    setModel(model);

  }


  const handle_prag = (e) => {
    setPrag(e.target.value);

  }

  const handle_glu = (e) => {
    setGlu(e.target.value);
  }

  const handle_blo = (e) => {
    setBlo(e.target.value);
  }

  const handle_ski = (e) => {
    setSki(e.target.value);
  }

  const handle_ins = (e) => {
    setIns(e.target.value);
  }

  const handle_bmi = (e) => {
    setBmi(e.target.value);
  }

  const handle_dia = (e) => {
    setDia(e.target.value);
  }

  const handle_age = (e) => {
    setAge(e.target.value);

  }

  const handleSubmit = (e) => {
    e.preventDefault();
    const pred = model.predict(tf.tensor2d([Number(prag), Number(glu), Number(blo), Number(ski), Number(ins), Number(bmi), Number(dia), Number(age)], [1, 8]));
    pred.print();
    
    pred.data().then(function (da) {
     let result = Array.from(da)
      let res = Math.round(result[0])
      
      if(res === 1){
        setOutput("Yes");
      }else{
        setOutput("No");
      }

    })

  }

  return (
    <div className="container">

      <div className="row">
        <div className="col-md-12">
          <p>Please Train the model least one time before the test</p>
          <button onClick={train} type="button" className="button">Start Training</button>
        </div>
      </div>


      <div className="row">

        <form onSubmit={handleSubmit}>

          <div className="form-group col-md-12">
            <label>Pregnancies:</label>
            <input type="number" value={prag} onChange={handle_prag} />
          </div>

          <div className="form-group col-md-12">
            <label> Glucose:</label>
              <input type="number" value={glu} onChange={handle_glu} />
          </div>

          <div className="form-group col-md-12">
            <label>BloodPressure:</label>
              <input type="number" value={blo} onChange={handle_blo} />
          </div>

           <div className="form-group col-md-12">
            <label>SkinThickness:</label>
              
            <input type="number" value={ski} onChange={handle_ski} />
          </div>

           <div className="form-group col-md-12">
            <label>Insulin:</label>
              
            <input type="number" value={ins} onChange={handle_ins} />
          </div>

           <div className="form-group col-md-12">
            <label>BMI:</label>
              
            <input type="number" value={bmi} onChange={handle_bmi} />
          </div>
           <div className="form-group col-md-12">
            <label>DiabetesPedigreeFunction:</label>
              
            <input type="number" value={dia} onChange={handle_dia} />

          </div>
           <div className="form-group col-md-12">
            <label>Age:</label>
            
            <input type="number" value={age} onChange={handle_age} />
          </div>

          <input type="submit" value="Test" className="button text-center"/>
      </form>

      </div>

      <p>Do I have diabetes?:  {output}</p>


    </div>
  );
}

export default App;
