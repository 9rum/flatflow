Graph {
    nodes: vec![
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EMBEDDING,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(50272, 0), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::ONES,
            args: vec![],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::FULL,
            args: vec![],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ARANGE,
            args: vec![],
            meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] }],
            meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::LT_TENSOR,
            args: vec![
                TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
                TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(0, 1), SymInt(1, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BOOL,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::MASKED_FILL_SCALAR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::RSUB_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BOOL,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::MASKED_FILL_SCALAR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::CUMSUM,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SUB_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EMBEDDING,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(2050, 0), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::ADDMM,
            args: vec![
                TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(5120, 0)] },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::MUL_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::MUL_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::BMM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_SOFTMAX,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::BMM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
            },
        },
        Node {
            target: Operator::ADDMM,
            args: vec![
                TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(20480, 0)] },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(20480, 0)],
            },
        },
        Node {
            target: Operator::RELU,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(20480, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(20480, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::ADDMM,
            args: vec![
                TensorMetadata { dtype: ScalarType::FLOAT16, shape: vec![SymInt(5120, 0)] },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(5120, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(50272, 0), SymInt(5120, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(5120, 0), SymInt(50272, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(50272, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(50272, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(0, 1), SymInt(50272, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(50272, 0)],
            },
        },
    ],
}
