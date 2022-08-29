use std::mem;
use std::os::raw::c_int;
use std::slice::from_raw_parts_mut;
use rand::Rng;

//pub mod mlp;
#[repr(C)]
#[derive(Debug)]
pub struct MLP {
    pub d: Vec<i32>,//npl
    pub length: usize,
    pub weight: Vec<Vec<Vec<f64>>>,//outputs
    pub x: Vec<Vec<f64>>,//inputs
    pub deltas: Vec<Vec<f64>>,//différences entre les inputs et les outputs

}

#[no_mangle]
pub extern "C" fn create_mlp(npl: *const i32, npl_size: i32) -> *mut MLP {
    let d = unsafe {
        std::slice::from_raw_parts(npl, npl_size as usize).to_vec()
    };

    let mut weight: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut x: Vec<Vec<f64>> = Vec::new();
    let mut deltas: Vec<Vec<f64>> = Vec::new();
    let mut rand = rand::thread_rng();

    for l in 0..d.len() {
        let w1: Vec<Vec<f64>> = Vec::new();
        weight.push(w1);

        if l == 0 {
            continue;
        }

        for i in 0..(d[l - 1] + 1) as usize {
            let w2: Vec<f64> = Vec::new();
            weight[l].push(w2);
            for _j in 0..d[l] + 1 {
                weight[l][i].push(rand.gen_range(-1.0, 1.0));
            }
        }
    }
    for l in 0..d.len() {
        let x1: Vec<f64> = Vec::new();
        let deltas1: Vec<f64> = Vec::new();
        x.push(x1);
        deltas.push(deltas1);

        for j in 0..d[l] + 1 {
            match j {
                0 => x[l].push(1.0),
                _ => x[l].push(0.0),
            }
            deltas[l].push(0.0);
        }
    }
    let model = Box::new(MLP {
        d: d.clone(),
        length: (d.len() - 1) as usize,
        weight,
        x,
        deltas,
    });

    let model_ptr = Box::leak(model);

    model_ptr
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_classification(model_ptr: *mut MLP, sample_inputs:*mut f64, input_size:i32) -> *mut f64{


    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    let sample_inputs = unsafe{
        from_raw_parts_mut(sample_inputs,input_size as usize)
    };

    classificationOrRegression(model, &sample_inputs.to_vec(),true);

    let mut result: Vec<f64> = vec![];

    for i in 1..(model.d[(model.length - 1) as usize]+1) as usize{
        result.push(model.x[(model.length - 1) as usize][i]); // result = model.x[(moc
    }

    let result_boxe = result.into_boxed_slice();
    let response = Box::leak(result_boxe);
    let end = response.as_mut_ptr();
    end
}

pub fn classificationOrRegression(model_ptr: *mut MLP, sample_inputs: &Vec<f64>, is_classification: bool) {
    let mut total = 0.0;

    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    for i in 1..model.d[0] as usize +1{
        model.x[0][i] = sample_inputs[i - 1];
    }
    for i in 1..model.d.len() as usize {
        for j in 1..(model.d[i] + 1) as usize {
            for k in 0..model.d[i - 1] as usize +1{
                total = total + model.weight[i][k][j] * model.x[i - 1][k]
            }
            model.x[i][j] = total;
            if (i as i32) < (model.d.len() - 1) as i32 || is_classification {//j'applique la tangente uniquement pour une classification
                model.x[i][j] = model.x[i][j].tanh();
            }
        }
    }
}

#[no_mangle]
pub extern "C" fn predict_mlp_model_regression(model_ptr: *mut MLP, sample_inputs:*mut f64, input_size:i32) -> *mut f64{

    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    let sample_inputs = unsafe{
        from_raw_parts_mut(sample_inputs,input_size as usize)
    };

    classificationOrRegression(model, &sample_inputs.to_vec(),false);

    let mut result: Vec<f64> = Vec::new();

    for i in 1..(model.d[(model.length -1) as usize]+1) as usize{
        result.push(model.x[(model.length - 1) as usize][i]);
    }

    let result_boxe = result.into_boxed_slice();
    let response = Box::leak(result_boxe);
    let res = response.as_mut_ptr();
    res
}

#[no_mangle]
pub extern "C" fn train_mlp(model_ptr: *mut MLP, all_samples_inputs: *mut f64, first_input_size:i32,
                            all_samples_expected_outputs: *mut f64, second_input_size:i32){

    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    let all_samples_inputs = unsafe{
        from_raw_parts_mut(all_samples_inputs,first_input_size as usize)
    };

    let all_samples_expected_outputs = unsafe{
        from_raw_parts_mut(all_samples_expected_outputs,second_input_size as usize)
    };

    let is_classification = true;
    let iterations_count = 1_000_000;
    let learning_rate = 0.01;

    let mut input_d = model.d[0];
    let mut output_d = model.d[(model.length-1) as usize];

    let mut sample_count = first_input_size / input_d;
    println!("{}", input_d);
    let mut sample_inputs:Vec<f64> = vec![];
    let mut total:f64;
    for i in 0..iterations_count{
        let mut k = rand::thread_rng().gen_range(0, sample_count) as usize;
        sample_inputs = all_samples_inputs[(k * input_d as usize)..((k + 1) * input_d as usize)].to_vec();

        let mut sample_expected_outputs = &all_samples_expected_outputs[(k * output_d as usize)..((k + 1) * output_d as usize)];
        classificationOrRegression(model, &sample_inputs,is_classification);

        for j in 1..model.d[(model.length -1) as usize] as usize +1{
            model.deltas[(model.length -1) as usize][j] = model.x[(model.length - 1) as usize][j] - sample_expected_outputs[j -1];
            if is_classification{
                model.deltas[(model.length -1) as usize][j] =
                    (1.0 - model.x[(model.length -1) as usize][j]
                        * model.x[(model.length -1) as usize][j])
                        * model.deltas[(model.length -1) as usize][j];
            }
        }

        for j in (1..model.length as usize).rev(){
            for x in 0..(model.d[j-1] +1 ) as usize{
                total = 0.0;
                for l in 1..(model.d[j]+1) as usize{
                    total += model.weight[j][x][l] * model.deltas[j][l];
                }
                model.deltas[j - 1][x] = (1.0 - model.x[j-1][x] * model.x[j-1][x]) * total;
            }
        }
        for j in 1..(model.length) as usize{
            for x in 0..(model.d[j-1] +1) as usize{
                for l in 1..(model.d[j] +1) as usize{
                    model.weight[j][x][l] -= learning_rate * model.x[j-1][x] * model.deltas[j][l];
                }
            }
        }
    }
}


/*#[no_mangle]
pub extern "C" fn train_mlp(model_ptr: *mut MLP, all_samples_inputs: *mut f64, first_input_size:i32,
                 all_samples_expected_outputs: *mut f64, second_input_size:i32) {
    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };

    let all_samples_inputs = unsafe{
        from_raw_parts_mut(all_samples_inputs,first_input_size as usize)
    };

    let all_samples_expected_outputs = unsafe{
        from_raw_parts_mut(all_samples_expected_outputs,second_input_size as usize)
    };


    let learning_rate = 0.001;
    let nb_iter = 1_000_000;
    let mut k;
    let mut input_d = model.d[0];
    let mut output_d = model.d[(model.length-1) as usize];
    let mut sample_inputs:Vec<f64> = vec![];
    let mut count = first_input_size / input_d;
    let is_classification = true;
    let mut rng = rand::thread_rng();



    for mut a in 0..nb_iter{
        k = rand::thread_rng().gen_range(0, count) as usize;
        sample_inputs = all_samples_inputs[(k*input_d as usize)..((k+1) * input_d as usize)].to_vec();
        let sample_expected_output = &all_samples_expected_outputs[(k * output_d as usize)..((k+1) * output_d as usize)];
        classificationOrRegression(model, &sample_inputs.to_vec(), is_classification);

        for j in 1..model.d[model.length - 1 as usize] as usize + 1{
            model.x[(model.length - 1) as usize][j] = model.x[(model.length - 1) as usize][j] - sample_expected_output[j-1];
            if is_classification {
                model.x[model.length as usize][j] = model.x[model.length as usize][j] * (1.0 - f64::powi(model.x[model.length][j], 2));
            }
            model.deltas[model.length][j] = model.x[model.length as usize][j];
        }
        for l in (1..model.length as usize + 1).rev(){
            for i in 1..(model.d[l-1] +1) as usize{
                let mut total = 0.0;
                for j in 1..(model.d[l] + 1) as usize{
                    total += model.weight[l][i][j] * model.deltas[l][j];
                }
                total = (1.0 - f64::powi(model.x[l - 1][i], 2)) * total;
                model.deltas[l-1][i] = total;
            }
        }
        for l in 1..model.length + 1{
            for i in 0..(model.d[l - 1] + 1) as usize{
                for j in 0..(model.d[l]+1) as usize{
                    model.weight[l][i][j] -= learning_rate * model.x[l - 1][i] * model.deltas[l][j];
                }
            }
        }
    }


}*/

/*#[no_mangle]
pub extern "C" fn train_mlp(model_ptr: *mut MLP, all_samples_inputs: *mut f64, first_input_size:i32,
                 all_samples_expected_outputs: *mut f64, second_input_size:i32){

    let model: &mut MLP = unsafe{ &mut *model_ptr.as_mut().unwrap() };
    let learning_rate = 0.001;
    let nb_iter = 10_000;
    let mut input_d = model.d[0];
    let mut output_d = model.d[(model.length-1) as usize];
    let is_classification = true;
    let mut rng = rand::thread_rng();
    let mut count =first_input_size / input_d;


    let all_samples_inputs = unsafe{
        from_raw_parts_mut(all_samples_inputs,first_input_size as usize)
    };

    let all_samples_expected_outputs = unsafe{
        from_raw_parts_mut(all_samples_expected_outputs,second_input_size as usize)
    };

    println!("{}", count);


    for mut a in 0..nb_iter{
        let mut k = rng.gen_range(0, count) as usize;
        let sample_inputs_res = all_samples_inputs[k.clone()];
        let sample_expected_output = &all_samples_expected_outputs[(k * output_d as usize)];
        let mut result: Vec<f64> = Vec::new();
        result.push(sample_inputs_res);
        let res = result.as_mut_ptr();
        predict_mlp_model_classification(model, res, first_input_size);

        for j in 1..(model.d[(model.length) as usize] as usize +1) as usize{
            let mut semi_gradient = (model.x[model.length as usize][j] - (sample_expected_output[j-1] as f64)) as f64;
            if is_classification {
                semi_gradient = semi_gradient * (1.0 - f64::powi(model.x[model.length][j], 2));
            }
            model.deltas[model.length][j] = semi_gradient;
        }
        for l in (1..model.length as usize + 1).rev(){
            for i in 1..(model.d[l-1] +1) as usize{
                let mut total = 0.0;
                for j in 1..(model.d[l] + 1) as usize{
                    total += model.weight[l][i][j] * model.deltas[l][j];
                }
                total = (1.0 - f64::powi(model.x[l - 1][i], 2)) * total;
                model.deltas[l-1][i] = total;
            }
        }
        for l in 1..model.length + 1{
            for i in 0..(model.d[l - 1] + 1) as usize{
                for j in 0..(model.d[l]+1) as usize{
                    model.weight[l][i][j] -= learning_rate * model.x[l - 1][i] * model.deltas[l][j];
                }
            }
        }
    }

}*/

/*#[no_mangle]
pub extern "C" fn destroy_mlp(model:*mut mlp){
    unsafe{
        let i = Box::from_raw(model);
    }
}*/

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}



