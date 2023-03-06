def load(path, **configs):
    '''
    Load an object can be used in paddle from specified path.

    Note:
        Now supports loading ``state_dict`` of Layer/Optimizer, Tensor and nested structure containing Tensor, Program.

    Note:
        In order to use the model parameters saved by paddle more efficiently,
        ``paddle.load`` supports loading ``state_dict`` of Layer from the result of
        other save APIs except ``paddle.save`` , but the argument ``path`` format is
        different:
        1. loading from ``paddle.static.save`` or ``paddle.Model().save(training=True)`` ,
        ``path`` needs to be a complete file name, such as ``model.pdparams`` or
        ``model.pdopt`` ;
        2. loading from ``paddle.jit.save`` or ``paddle.static.save_inference_model``
        or ``paddle.Model().save(training=False)`` , ``path`` need to be a file prefix,
        such as ``model/mnist``, and ``paddle.load`` will get information from
        ``mnist.pdmodel`` and ``mnist.pdiparams`` ;
        3. loading from paddle 1.x APIs ``paddle.fluid.io.save_inference_model`` or
        ``paddle.fluid.io.save_params/save_persistables`` , ``path`` need to be a
        directory, such as ``model`` and model is a directory.

    Note:
        If you load ``state_dict`` from the saved result of static graph mode API such as
        ``paddle.static.save`` or ``paddle.static.save_inference_model`` ,
        the structured variable name in dynamic mode will cannot be restored.
        You need to set the argument ``use_structured_name=False`` when using
        ``Layer.set_state_dict`` later.

    Args:
        path(str|BytesIO) : The path/buffer to load the target object. Generally, the path is the target
            file path. When loading state_dict from the saved result of the API used to save
            the inference model, the path may be a file prefix or directory.
        **configs (dict, optional): other load configuration options for compatibility. We do not
            recommend using these configurations, they may be removed in the future. If not necessary,
            DO NOT use them. Default None.
            The following options are currently supported:
            (1) model_filename (str): The inference model file name of the paddle 1.x
            ``save_inference_model`` save format. Default file name is :code:`__model__` .
            (2) params_filename (str): The persistable variables file name of the paddle 1.x
            ``save_inference_model`` save format. No default file name, save variables separately
            by default.
            (3) return_numpy(bool): If specified as True, return tensor as numpy.ndarray, otherwise return tensor as paddle.Tensor.
            Default False.

    Returns:
        Object(Object): a target object can be used in paddle

    Examples:
        .. code-block:: python
            :name: code-example-1

            # example 1: dynamic graph
            import paddle
            emb = paddle.nn.Embedding(10, 10)
            layer_state_dict = emb.state_dict()

            # save state_dict of emb
            paddle.save(layer_state_dict, "emb.pdparams")

            scheduler = paddle.optimizer.lr.NoamDecay(
                d_model=0.01, warmup_steps=100, verbose=True)
            adam = paddle.optimizer.Adam(
                learning_rate=scheduler,
                parameters=emb.parameters())
            opt_state_dict = adam.state_dict()

            # save state_dict of optimizer
            paddle.save(opt_state_dict, "adam.pdopt")
            # save weight of emb
            paddle.save(emb.weight, "emb.weight.pdtensor")

            # load state_dict of emb
            load_layer_state_dict = paddle.load("emb.pdparams")
            # load state_dict of optimizer
            load_opt_state_dict = paddle.load("adam.pdopt")
            # load weight of emb
            load_weight = paddle.load("emb.weight.pdtensor")

        .. code-block:: python
            :name: code-example-2

            # example 2: Load multiple state_dict at the same time
            import paddle
            from paddle import nn
            from paddle.optimizer import Adam

            layer = paddle.nn.Linear(3, 4)
            adam = Adam(learning_rate=0.001, parameters=layer.parameters())
            obj = {'model': layer.state_dict(), 'opt': adam.state_dict(), 'epoch': 100}
            path = 'example/model.pdparams'
            paddle.save(obj, path)
            obj_load = paddle.load(path)

        .. code-block:: python
            :name: code-example-3

            # example 3: static graph
            import paddle
            import paddle.static as static

            paddle.enable_static()

            # create network
            x = paddle.static.data(name="x", shape=[None, 224], dtype='float32')
            z = paddle.static.nn.fc(x, 10)

            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            exe.run(paddle.static.default_startup_program())
            prog = paddle.static.default_main_program()
            for var in prog.list_vars():
                if list(var.shape) == [224, 10]:
                    tensor = var.get_value()
                    break

            # save/load tensor
            path_tensor = 'temp/tensor.pdtensor'
            paddle.save(tensor, path_tensor)
            load_tensor = paddle.load(path_tensor)

            # save/load state_dict
            path_state_dict = 'temp/model.pdparams'
            paddle.save(prog.state_dict("param"), path_tensor)
            load_state_dict = paddle.load(path_tensor)

        .. code-block:: python
            :name: code-example-4

            # example 4: load program
            import paddle

            paddle.enable_static()

            data = paddle.static.data(
                name='x_static_save', shape=(None, 224), dtype='float32')
            y_static = z = paddle.static.nn.fc(data, 10)
            main_program = paddle.static.default_main_program()
            path = "example/main_program.pdmodel"
            paddle.save(main_program, path)
            load_main = paddle.load(path)
            print(load_main)

        .. code-block:: python
            :name: code-example-5

            # example 5: save object to memory
            from io import BytesIO
            import paddle
            from paddle.nn import Linear
            paddle.disable_static()

            linear = Linear(5, 10)
            state_dict = linear.state_dict()
            byio = BytesIO()
            paddle.save(state_dict, byio)
            tensor = paddle.randn([2, 3], dtype='float32')
            paddle.save(tensor, byio)
            byio.seek(0)
            # load state_dict
            dict_load = paddle.load(byio)

    '''

    if _is_memory_buffer(path) or os.path.isfile(path):
        config = _parse_load_config(configs)
        exception_type = pickle.UnpicklingError
        try:
            with _open_file_buffer(path, 'rb') as f:
                # When value of dict is lager than 4GB ,there is a Bug on 'MAC python3'
                if (
                    _is_file_path(path)
                    and sys.platform == 'darwin'
                    and sys.version_info.major == 3
                ):
                    load_result = _pickle_loads_mac(path, f)
                else:
                    load_result = pickle.load(f, encoding='latin1')

                # TODO(weixin):If `obj` is any object, the judgment condition should be more precise.
                if isinstance(load_result, dict):
                    load_result = _pack_loaded_dict(load_result)
                    # paddle2.0: paddle.save/load
                    if "StructuredToParameterName@@" in load_result:

                        for key in load_result["StructuredToParameterName@@"]:
                            if isinstance(load_result[key], np.ndarray):
                                load_result[key] = _ndarray_to_tensor(
                                    load_result[key], config.return_numpy
                                )

                        if (
                            not config.keep_name_table
                            and "StructuredToParameterName@@" in load_result
                        ):
                            del load_result["StructuredToParameterName@@"]
                    else:
                        # paddle2.1 static.save/load
                        load_result = _parse_load_result(
                            load_result, config.return_numpy
                        )

                else:
                    load_result = _parse_load_result(
                        load_result, config.return_numpy
                    )

        except exception_type as msg_pickle:
            try:
                tensor, _ = _load_selected_rows(path)
                return tensor
            except:
                try:
                    tensor, _ = _load_lod_tensor(path)
                    if config.return_numpy:
                        return np.array(tensor)
                    else:
                        if _non_static_mode():
                            return _lod_tensor2varbase(tensor)
                        return tensor
                except:
                    try:
                        with _open_file_buffer(path, "rb") as f:
                            program_desc_str = f.read()
                            program = Program.parse_from_string(
                                program_desc_str
                            )
                            return program
                    except:
                        raise ValueError(
                            "`paddle.load` can not parse the file:{}.".format(
                                path
                            )
                        )

    else:
        load_result = _legacy_load(path, **configs)

    return load_result


def _legacy_load(path, **configs):
    load_result = None
    config = _parse_load_config(configs)

    if os.path.isfile(path) or _is_memory_buffer(path):
        # we think path is file means this file is created by paddle.save
        with _open_file_buffer(path, 'rb') as f:
            load_result = pickle.load(f, encoding='latin1')
        load_result = _pack_loaded_dict(load_result)
        if (
            not config.keep_name_table
            and "StructuredToParameterName@@" in load_result
        ):
            del load_result["StructuredToParameterName@@"]
    else:
        # file prefix and directory are compatible cases
        model_path, config = _build_load_path_and_config(path, config)
        # check whether model file exists
        if config.model_filename is None:
            model_filename = '__model__'
        else:
            model_filename = config.model_filename
        model_file_path = os.path.join(model_path, model_filename)

        if os.path.exists(model_file_path):
            # Load state dict by `jit.save/io.save_inference_model` save format
            # NOTE(chenweihang): [ Compatibility of save_inference_model save format ]
            # The model saved by `save_inference_model` does not completely correspond to
            # the information required by the `state_dict` under the dygraph.
            # `save_inference_model` not save structured name, we need to remind
            # the user to configure the `use_structured_name` argument when `set_state_dict`
            # NOTE(chenweihang): `jit.save` doesn't save optimizer state
            load_result = _load_state_dict_from_save_inference_model(
                model_path, config
            )
        else:
            # load state dict by `io.save_params/persistables` save format
            # TODO(chenweihang): [ Now only supports loading parameters separately ]
            # If users save all parameters as one file, the [ variable.name -> variable ]
            # mapping info will lost, so users need to give variable list, but users build
            # variable list in dygraph mode is difficult, we recommend users to use
            # paddle.static.load_program_state in this case
            load_result = _load_state_dict_from_save_params(model_path)

    return load_result
