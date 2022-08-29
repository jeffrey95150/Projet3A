import ctypes

import numpy as np


def test(my_dll):

    npl = [2, 2, 1]

    arr_type_npl = ctypes.c_int32 * len(npl)

    my_dll.create_mlp.argtypes = [arr_type_npl, ctypes.c_int32]
    my_dll.create_mlp.restype = ctypes.c_void_p

    model = my_dll.create_mlp(arr_type_npl(*npl), len(npl))

    print("--------------------")

    X = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    Y = np.array([
        [-1.0],
        [1.0],
        [1.0],
        [1.0],
    ])

    A = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    B = np.array([-1.0, 1.0, 1.0, 1.0])

    arr_type_inputs = ctypes.c_double * len(X)
    arr_type_outputs = ctypes.c_double * len(Y)

    arr_type_inputs_res = ctypes.c_double * len(A)
    arr_type_outputs_res = ctypes.c_double * len(B)

    my_dll.predict_mlp_model_classification.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    my_dll.predict_mlp_model_classification.restype = ctypes.POINTER(ctypes.c_double)

    my_dll.predict_mlp_model_regression.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    my_dll.predict_mlp_model_regression.restype = ctypes.POINTER(ctypes.c_double)

    print("predict_mlp")
    for i in X:
        native_result = my_dll.predict_mlp_model_classification(model, arr_type_inputs(*i), 2)
        result = np.ctypeslib.as_array(native_result, (1,))
        print(result)
        # my_dll.destroy_float_array(native_result)

    """for i in X:
        native_result = my_dll.predict_mlp_model_regression(model, arr_type_inputs(*i), 2)
        result = np.ctypeslib.as_array(native_result, (1,))
        print(result)"""

    print(len(A))
    print(len(B))

    my_dll.train_mlp.argtypes = [ctypes.c_void_p, arr_type_inputs_res, ctypes.c_int, arr_type_outputs_res, ctypes.c_int]
    my_dll.train_mlp.restype = None

    my_dll.train_mlp(model, arr_type_inputs_res(*A), len(A), arr_type_outputs_res(*B), len(B))

    print("--------------------")
    print("AFTER TRAINING")
    print("--------------------")

    for i in X:
        native_result = my_dll.predict_mlp_model_classification(model, arr_type_inputs(*i), len(i))
        result = np.ctypeslib.as_array(native_result, (1,))
        print(result)


if __name__ == '__main__':
    my_rust_dll = ctypes.CDLL(
        "C:/Users/jeffr/CLionProjects/PMCpa/target/release/PMCpa.dll")

    print("In Rust :")
    test(my_rust_dll)
