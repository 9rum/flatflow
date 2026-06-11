// SPDX-License-Identifier: Apache-2.0

use flatbuffers::{FlatBufferBuilder, InvalidFlatbuffer};

use crate::ops::graph::{Graph, Node, SymInt, TensorMetadata};
use crate::ops::graph_generated::{self, root_as_graph};
use crate::ops::operator_generated::Operator;
use crate::ops::scalar_type_generated::ScalarType;
use crate::ops::transform;

#[test]
fn test_transform_with_mistral3() -> Result<(), InvalidFlatbuffer> {
    // This graph has been generated based on the exported [Mistral Small 3]:
    // * torch        2.4.0a0+3bcc3cddb5.nv24.7
    // * transformers 4.51.3
    //
    // [Mistral Small 3]: https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501
    let mistral3 = Graph {
        nodes: vec![
            Node {
                target: Operator::EMBEDDING,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(131072, 0), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
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
                    shape: vec![SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::ARANGE,
                args: vec![],
                meta: TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(1, 1)] },
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
                    TensorMetadata { dtype: ScalarType::INT64, shape: vec![SymInt(1, 1)] },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(0, 1), SymInt(1, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BOOL,
                    shape: vec![SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(1, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BOOL,
                        shape: vec![SymInt(0, 1), SymInt(1, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(64, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
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
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
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
                        shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(1, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(64, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::CAT,
                args: vec![],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            // Node {
            //     target: Operator::COS,
            //     args: vec![TensorMetadata {
            //         dtype: ScalarType::FLOAT32,
            //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            //     }],
            //     meta: TensorMetadata {
            //         dtype: ScalarType::FLOAT32,
            //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            //     },
            // },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            // Node {
            //     target: Operator::SIN,
            //     args: vec![TensorMetadata {
            //         dtype: ScalarType::FLOAT32,
            //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            //     }],
            //     meta: TensorMetadata {
            //         dtype: ScalarType::FLOAT32,
            //         shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
            //     },
            // },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::POW_TENSOR_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::MEAN_DIM,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
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
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::_TO_COPY,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata { dtype: ScalarType::BFLOAT16, shape: vec![SymInt(5120, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4096, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(4096, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(4096, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(4096, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(4096, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4096, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4096, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32, 0), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1024, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(1024, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(1024, 0)],
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
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(8, 0), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(8, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::NEG,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(64, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::NEG,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(64, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(64, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(8, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(1, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(1, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(1, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(1, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(4, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(4, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(4, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                },
            },
            Node {
                target: Operator::_UNSAFE_VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![
                        SymInt(1, 0),
                        SymInt(8, 0),
                        SymInt(4, 0),
                        SymInt(0, 1),
                        SymInt(128, 0),
                    ],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(1, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(1, 1)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(32, 0), SymInt(128, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_SOFTMAX,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(32, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32, 0), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(4096, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(4096, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(4096, 0), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(4096, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(4096, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32768, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(32768, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(32768, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::SILU,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(32768, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(32768, 0), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(32768, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(32768, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(32768, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(32768, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::T,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(131072, 0), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(5120, 0), SymInt(131072, 0)],
                },
            },
            Node {
                target: Operator::MM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::BFLOAT16,
                        shape: vec![SymInt(5120, 0), SymInt(131072, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(131072, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(0, 1), SymInt(131072, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::BFLOAT16,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(131072, 0)],
                },
            },
        ],
    };

    let mut builder = FlatBufferBuilder::new();
    let mut nodes = Vec::new();
    for node in mistral3.nodes.iter() {
        let mut args = Vec::new();
        for arg in node.args.iter() {
            let mut shape = Vec::new();
            for int in arg.shape.iter() {
                shape.push(graph_generated::SymInt::new(&[int.0, int.1]));
            }
            let shape = Some(builder.create_vector(shape.as_slice()));

            args.push(graph_generated::TensorMetadata::create(
                &mut builder,
                &graph_generated::TensorMetadataArgs { dtype: arg.dtype, shape },
            ));
        }
        let args = Some(builder.create_vector(args.as_slice()));

        let mut shape = Vec::new();
        for int in node.meta.shape.iter() {
            shape.push(graph_generated::SymInt::new(&[int.0, int.1]));
        }
        let shape = Some(builder.create_vector(shape.as_slice()));

        let meta = Some(graph_generated::TensorMetadata::create(
            &mut builder,
            &graph_generated::TensorMetadataArgs { dtype: node.meta.dtype, shape },
        ));

        nodes.push(graph_generated::Node::create(
            &mut builder,
            &graph_generated::NodeArgs { target: node.target, args, meta },
        ));
    }
    let nodes = Some(builder.create_vector(nodes.as_slice()));

    let graph = graph_generated::Graph::create(&mut builder, &graph_generated::GraphArgs { nodes });
    builder.finish(graph, None);

    let proj = transform(root_as_graph(builder.finished_data())?); // 8417 s0^2 + 1054055301 s0
    assert_eq!(proj(0), Ok(0));
    assert_eq!(proj(1), Ok(1054063718));
    assert_eq!(proj(1024), Ok(1088178492416));
    assert_eq!(proj(2048), Ok(2194008713216));

    Ok(())
}
