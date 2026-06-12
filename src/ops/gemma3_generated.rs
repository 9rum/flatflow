Graph {
    nodes: vec![
        Node {
            target: Operator::EMBEDDING,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(262144, 0), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::INT64,
                    shape: vec![SymInt(1, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![] }],
            meta: TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![] },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![] },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::ARANGE_START,
            args: vec![],
            meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(0, 1)] }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::FULL,
            args: vec![],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::TRIU,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ARANGE,
            args: vec![],
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
            target: Operator::GT_TENSOR,
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
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
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
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::INT64,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::BMM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(1, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(128, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::CAT,
            args: vec![],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        // Node {
        //     target: Operator::COS,
        //     args: vec![TensorMetadata {
        //         dtype: ScalarType::FLOAT32,
        //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
        //     }],
        //     meta: TensorMetadata {
        //         dtype: ScalarType::FLOAT32,
        //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
        //     },
        // },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        // Node {
        //     target: Operator::SIN,
        //     args: vec![TensorMetadata {
        //         dtype: ScalarType::FLOAT32,
        //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
        //     }],
        //     meta: TensorMetadata {
        //         dtype: ScalarType::FLOAT32,
        //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
        //     },
        // },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ONES_LIKE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BOOL,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::TRIL,
            args: vec![TensorMetadata {
                dtype: ScalarType::BOOL,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BOOL,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::SCALAR_TENSOR,
            args: vec![],
            meta: TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![] },
        },
        Node {
            target: Operator::WHERE_SELF,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![] },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::POW_TENSOR_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::MEAN_DIM,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::RSQRT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0)],
            }],
            meta: TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(1152, 0)] },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1152, 0)],
            }],
            meta: TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(1152, 0)] },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(1152, 0)] },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1024, 0), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(1024, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1152, 0), SymInt(1024, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1024, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1024, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1024, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1024, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4, 0), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4, 0), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(256, 0), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1152, 0), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::POW_TENSOR_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MEAN_DIM,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::RSQRT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(1, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(256, 0)],
            }],
            meta: TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(256, 0)] },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(256, 0)],
            }],
            meta: TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(256, 0)] },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(256, 0)] },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::POW_TENSOR_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MEAN_DIM,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::RSQRT,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(256, 0)] },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::FLOAT32,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::NEG,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::NEG,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::_TO_COPY,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::UNSQUEEZE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(4, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![
                    SymInt(1, 0),
                    SymInt(1, 0),
                    SymInt(4, 0),
                    SymInt(0, 1),
                    SymInt(256, 0),
                ],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::MUL_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::TRANSPOSE_INT,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::MUL_SCALAR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::BMM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4, 0), SymInt(256, 0), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::_SOFTMAX,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::EXPAND,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
            },
        },
        Node {
            target: Operator::BMM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(0, 1)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(4, 0), SymInt(0, 1), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::CLONE,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4, 0), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4, 0), SymInt(256, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4, 0), SymInt(256, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1024, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(1024, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1024, 0), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1024, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1024, 0), SymInt(1152, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::ADD_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(6912, 0), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1152, 0), SymInt(6912, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(6912, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::GELU,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::MUL_TENSOR,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(6912, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(6912, 0), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(6912, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(6912, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(6912, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(6912, 0), SymInt(1152, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::SLICE_TENSOR,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1152, 0)],
            },
        },
        Node {
            target: Operator::T,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(262144, 0), SymInt(1152, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1152, 0), SymInt(262144, 0)],
            },
        },
        Node {
            target: Operator::MM,
            args: vec![
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1152, 0)],
                },
                TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1152, 0), SymInt(262144, 0)],
                },
            ],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(262144, 0)],
            },
        },
        Node {
            target: Operator::VIEW,
            args: vec![TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(0, 1), SymInt(262144, 0)],
            }],
            meta: TensorMetadata {
                dtype: ScalarType::BFLOAT16,
                shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(262144, 0)],
            },
        },
    ],
}
