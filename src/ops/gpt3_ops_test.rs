// SPDX-License-Identifier: Apache-2.0

use flatbuffers::{FlatBufferBuilder, InvalidFlatbuffer};

use crate::ops::graph::{Graph, Node, SymInt, TensorMetadata};
use crate::ops::graph_generated::{self, root_as_graph};
use crate::ops::operator_generated::Operator;
use crate::ops::scalar_type_generated::ScalarType;
use crate::ops::transform;

#[test]
fn test_transform_with_gpt3() -> Result<(), InvalidFlatbuffer> {
    // This graph has been generated based on the exported [GPT-3] via [torch.export.export]:
    // * torch        2.4.0a0+3bcc3cddb5.nv24.7
    // * transformers 4.46.2
    //
    // Note that the original GPT-3 models have thousands of nodes when converted to computational
    // graphs; this produces hundreds of thousands of lines of code when generated, severely slowing
    // down the build. To this end, this test emulates GPT-3 where an unique combination of
    // operator, data types and symbolic shapes appears only once, limiting the computational graph
    // to have only 52 nodes.
    //
    // [GPT-3]: https://huggingface.co/openai-community/gpt2
    // [torch.export.export]: https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/export.html
    let gpt3 = Graph {
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
                target: Operator::EMBEDDING,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(50257, 0), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::EMBEDDING,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(2048, 0), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::INT64,
                        shape: vec![SymInt(1, 0), SymInt(0, 1)],
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
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
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
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::UNSQUEEZE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::SLICE_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(15360, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(5120, 0), SymInt(15360, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(15360, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(15360, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(15360, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::TRANSPOSE_INT,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::MUL_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(40, 0), SymInt(128, 0), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(1, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::_SOFTMAX,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::EXPAND,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                },
            },
            Node {
                target: Operator::BMM,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(0, 1)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(40, 0), SymInt(0, 1), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::CLONE,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(40, 0), SymInt(128, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(5120, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(5120, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(5120, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(20480, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(5120, 0), SymInt(20480, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::POW_TENSOR_SCALAR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::TANH,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::ADD_TENSOR,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::MUL_TENSOR,
                args: vec![
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::VIEW,
                args: vec![TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(1, 0), SymInt(0, 1), SymInt(20480, 0)],
                }],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                },
            },
            Node {
                target: Operator::ADDMM,
                args: vec![
                    TensorMetadata { dtype: ScalarType::FLOAT32, shape: vec![SymInt(5120, 0)] },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(0, 1), SymInt(20480, 0)],
                    },
                    TensorMetadata {
                        dtype: ScalarType::FLOAT32,
                        shape: vec![SymInt(20480, 0), SymInt(5120, 0)],
                    },
                ],
                meta: TensorMetadata {
                    dtype: ScalarType::FLOAT32,
                    shape: vec![SymInt(0, 1), SymInt(5120, 0)],
                },
            },
        ],
    };

    let mut builder = FlatBufferBuilder::new();
    let mut nodes = Vec::new();
    for node in gpt3.nodes.iter() {
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

    let expr = transform(root_as_graph(builder.finished_data())?); // 1315 s0^2 + 39372164 s0
    assert_eq!(expr(0), Ok(0));
    assert_eq!(expr(1), Ok(39373479));
    assert_eq!(expr(1024), Ok(41695973376));
    assert_eq!(expr(2048), Ok(86149701632));

    Ok(())
}
