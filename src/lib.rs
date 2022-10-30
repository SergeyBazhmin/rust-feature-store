use std::collections::HashMap;

use pyo3::exceptions::{PyConnectionError, PyValueError};
use pyo3::prelude::*;
use polars::prelude::*;
use polars::export::arrow::ffi as ffi;
use polars_arrow::prelude::ArrayRef;
use pyo3::ffi::Py_uintptr_t;
use pyo3::types::{PyList, PyDict};
use redis::{self, Commands, RedisResult};



fn to_py_array(py: Python, pyarrow: &PyModule, array: ArrayRef) -> PyResult<PyObject> {
    let schema = Box::new(ffi::export_field_to_c(&ArrowField::new(
        "",
        array.data_type().clone(),
        true,
    )));
    let array = Box::new(ffi::export_array_to_c(array));

    let schema_ptr: *const ffi::ArrowSchema = &*schema;
    let array_ptr: *const ffi::ArrowArray = &*array;

    let array = pyarrow.getattr("Array")?.call_method1(
        "_import_from_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    Ok(array.to_object(py))
}

fn array_to_rust(arrow_array: &PyAny) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    arrow_array.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).unwrap();
        let array = ffi::import_array_from_c(*array, field.data_type).unwrap();
        Ok(array)
    }
}

pub fn py_series_to_rust_series(series: &PyAny) -> PyResult<Series> {
    // rechunk series so that they have a single arrow array
    let series = series.call_method0("rechunk")?;

    let name = series.getattr("name")?.extract::<String>()?;

    // retrieve pyarrow array
    let array = series.call_method0("to_arrow")?;
    // retrieve rust arrow array
    let array = array_to_rust(array)?;

    Series::try_from((name.as_str(), array)).map_err(|e| PyValueError::new_err(format!("{}", e)))
}

fn py_df_to_rust_dataframe(py_df: &PyAny) -> PyResult<DataFrame> {
    let pydf_columns = py_df.getattr("columns")?.cast_as::<PyList>()?;
    let mut rust_series: Vec<Series> = Vec::new();
    for col in pydf_columns {
        rust_series.push(py_series_to_rust_series(py_df.get_item(col)?)?);
    }
    DataFrame::new(rust_series).map_err(|e| PyValueError::new_err(e.to_string()))
}


fn series_vector_to_pydf(py: Python, series_vec: Vec<Series>) -> PyResult<PyObject> {
    let pyarrow = py.import("pyarrow")?;

    let arrow_arrays = series_vec.iter().map(|s| s.rechunk().to_arrow(0));
    let array_names: Vec<&str> = series_vec.iter().map(|s| s.name()).collect();

    let mut pyarrow_arrays = Vec::new();
    for arr in arrow_arrays {
        pyarrow_arrays.push(to_py_array(py, pyarrow, arr)?);
    }

    let pylist_of_arrays = PyList::new(py, pyarrow_arrays);
    let pylist_names = PyList::new(py, array_names);

    let arrow_table = pyarrow.getattr("Table")?.call_method1(
        "from_arrays", (pylist_of_arrays, pylist_names)
    )?.to_object(py);

    let polars = py.import("polars")?;
    let out = polars.call_method1(
        "from_arrow",
         (arrow_table,)
    )?;

    Ok(out.to_object(py))
}


pub fn string_to_polars_dtype(s: &str) -> DataType {
    match s {
        "i8" => DataType::Int8,
        "i16" => DataType::Int16,
        "i32" => DataType::Int32,
        "i64" => DataType::Int64,
        "f32" => DataType::Float32,
        "f64" => DataType::Float64,
        "utf8" => DataType::Utf8,
        _ => panic!("unknown type")
    }
}


#[pyclass]
struct FeatureStoreClient {
    connection: redis::Connection,
    table_meta: HashMap<String, Vec<String>>,
    // datasets: HashMap<String, HashMap<String, Vec<String>>>
}

#[pymethods]
impl FeatureStoreClient {
    #[new]
    fn new(host: String, port: i32) -> PyResult<Self> {
        let client = redis::Client::open(format!("redis://{host}:{port}/")).unwrap();
        match client.get_connection() {
            Ok(conn) => Ok(Self {
                connection: conn,
                table_meta: HashMap::new(),
                // datasets: HashMap::new()
            }),
            Err(msg) => Err(PyValueError::new_err(msg.to_string()))
        }
   }

    fn get_tables(&mut self, py: Python) -> PyResult<PyObject> {
        let res: RedisResult<Vec<String>> = self.connection.smembers("tables");
        match res {
            Ok(vec) => Ok(PyList::new(py, vec).to_object(py)),
            Err(msg) => Err(PyValueError::new_err(msg.to_string())),
        }
    }

    // fn get_datasets(&mut self, py: Python) -> PyResult<PyObject> {
    //     let res: RedisResult<Vec<String>> = self.connection.lock().unwrap().smembers("datasets");
    //     match res {
    //         Ok(vec) => Ok(PyList::new(py, vec).to_object(py)),
    //         Err(msg) => Err(PyValueError::new_err(msg.to_string())),
    //     }
    // }

    // fn create_dataset(&mut self, py: Python, dataset_name: &PyAny, tables: &PyDict) -> PyResult<()> {
    //     let json = py.import("json")?;
    //     let payload_jstring = json.call_method1(
    //         "dumps", (tables,)
    //     )?.extract::<String>()?;
    //     let dataset_name_rstring = dataset_name.extract::<String>()?;
    //     let tables_hashmap = tables.extract::<HashMap<String, Vec<String>>>()?;

    //     let mut guard = self.connection.lock().unwrap();
    //     guard.sadd::<_, _, ()>("datasets".to_string(), dataset_name_rstring.clone()).map_err(
    //         |e| PyValueError::new_err(e.to_string())
    //     )?;
    //     guard.set::<_, _, ()>(format!("{dataset_name_rstring}_dataset"), payload_jstring).map_err(
    //         |e| PyValueError::new_err(e.to_string())
    //     )?;
    //     self.datasets.insert(dataset_name_rstring, tables_hashmap);
    //     Ok(())
    // }

    fn load_table_meta(&mut self, table_name: String) -> PyResult<()> {
        let mut pipe = redis::pipe();
        let res: Vec<Vec<String>> = pipe.lrange(
            format!("{table_name}_dtypes"), 0, -1
        ).lrange(
            format!("{table_name}_columns"), 0, -1
        ).lrange(
            format!("{table_name}_pkey"), 0, -1
        ).query( &mut self.connection).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let field_names = vec![
            format!("{table_name}_dtypes"), format!("{table_name}_columns"), format!("{table_name}_pkey")
        ];

        for (items, key) in res.into_iter().zip(field_names.into_iter()) {
            self.table_meta.insert(key, items);
        }

        Ok(())
    }

    fn create_table(
        &mut self, 
        table_name: String,
        columns: Vec<String>,
        primary_key: Vec<String>,
        data_types: Vec<String>
    ) -> PyResult<()> {
        let table_exists: bool = self.connection.sismember("tables", &table_name).map_err(
            |_| PyValueError::new_err("failed to connect to redis".to_string())
        )?;
        if table_exists {
            Err(PyValueError::new_err(format!("table {table_name} already exists")))
        } else {
            let available_types = ["i8", "i16", "i32", "i64", "f32", "f64", "utf8"];
            if primary_key.iter().any(|key_col| !columns.contains(key_col)) {
                Err(PyValueError::new_err("primary key columns are not present in columns argument"))
            } else if data_types.len() != columns.len() {
                Err(PyValueError::new_err("type list length != columns list length"))
            } else if data_types.iter().any(|dtype| !available_types.iter().any(|s| s == dtype)) {
                Err(PyValueError::new_err("Unknown dtype in data_type list, should be one of (i8, i16, i32, i64, f32, f64, utf8)"))
            } else {
                let res: RedisResult<()> = redis::pipe()
                    .sadd("tables", &table_name)
                    .rpush(format!("{table_name}_columns"), &columns)
                    .rpush(format!("{table_name}_pkey"), &primary_key)
                    .rpush(format!("{table_name}_dtypes"), &data_types)
                    .query(&mut self.connection);
                match res {
                    Ok(()) => {
                        self.table_meta.insert(format!("{table_name}_dtypes"), columns);
                        self.table_meta.insert(format!("{table_name}_columns"), primary_key);
                        self.table_meta.insert(format!("{table_name}_pkey"), data_types);
                        Ok(())
                    },
                    _ => Err(PyValueError::new_err("asd"))
                }
            }
        }
    }

    // fn describe_dataset(&mut self, py: Python, dataset_name: String) -> PyResult<PyObject> {
    //     let payload: String = self.connection.lock().unwrap().get(format!("{dataset_name}_dataset")).map_err(
    //         |e| PyValueError::new_err(e.to_string())
    //     )?;
    //     let mjson = py.import("json")?;
    //     Ok(mjson.call_method1("loads", (payload,))?.to_object(py))
    // }

    fn describe_table(&mut self, py: Python, table_name: String) -> PyResult<PyObject> {
        let table_pkey: Vec<String> = self.connection.lrange(
            format!("{table_name}_pkey"), 0, -1
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let table_columns: Vec<String> = self.connection.lrange(
            format!("{table_name}_columns"), 0, -1
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dtypes: Vec<String> = self.connection.lrange(
            format!("{table_name}_dtypes"), 0, -1
        ).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let info = PyDict::new(py);
        for (column_name, column_dtype) in table_columns.iter().zip(dtypes.iter()) {
            info.set_item(column_name, column_dtype)?;
        }
        info.set_item("primary_key", table_pkey)?;
        Ok(info.to_object(py))
    }

    fn select(&mut self, py: Python, table_keys: &PyDict, mut feature_columns: HashMap<String, Vec<String>>) -> PyResult<PyObject> {
        let mut rtable_keys: Vec<(String, DataFrame)> = Vec::new();

        for (name, keys)  in table_keys {
            let r_name = name.extract::<String>()?;
            let rdf = py_df_to_rust_dataframe(keys)?;
            rtable_keys.push((r_name, rdf));
        }

        let queries: Vec<(String, DataFrame, Vec<String>)> = rtable_keys.into_iter().map(
            move |(name, df)| 
            feature_columns.remove(&name).map(move |fcolumns| (name, df, fcolumns)).expect("feature_columns keys do not match with table_keys")
        ).collect();

        let mut pipe = redis::pipe();
        for (name, df, fcols) in queries.iter() {
            let pkey = format!("{name}_pkey");
            let registered_table_pkey: &Vec<String> = self.table_meta.get(&pkey).unwrap();

            let key_expr = registered_table_pkey.iter().map(
                |key| concat_str([lit(key.clone()), col(key).cast(DataType::Utf8)], "=")
            ).collect::<Vec<Expr>>();

            let keys_df = df.clone().lazy().select([
                concat_str(key_expr, ";").alias("table_key")
            ]).collect().unwrap();
    
            let keys = keys_df.column("table_key").unwrap().rechunk();
            for key in keys.iter() {
                let key_str = match key {
                    AnyValue::Utf8(val) => val,
                    _ => panic!("expected string key")
                };
                pipe.hget(key_str, fcols);
            }
        }
        let col_values: Vec<Vec<Option<String>>> = pipe.query(&mut self.connection).unwrap();
        let mut offset: usize = 0;

        let feature_dict = PyDict::new(py);
        for (name, df, fcols) in queries.into_iter() {
            let ckey = format!("{name}_columns");
            let dkey = format!("{name}_dtypes");

            let dtypes: &Vec<String> = self.table_meta.get(&dkey).unwrap();
            let registered_table_columns: &Vec<String> = self.table_meta.get(&ckey).unwrap();
            let dtype_map: HashMap<String, String> = registered_table_columns.iter().cloned().zip(dtypes.iter().cloned()).collect();
    
            let (chunk_length, _) = df.shape();
            let chunk_col_values = &col_values[offset..offset + chunk_length];

            let mut feature_series: Vec<Series> = Vec::new();
            for (idx, column) in fcols.iter().enumerate() {
                let single_col_values: Vec<Option<&str>> = chunk_col_values.iter().map(|s| s[idx].as_deref()).collect();
                let inner_dtype = dtype_map.get(column).unwrap();
                let polars_dtype = string_to_polars_dtype(inner_dtype);
                let series = Series::new(column, single_col_values).strict_cast(&polars_dtype).unwrap();
                feature_series.push(series);
            }
            feature_dict.set_item(name, series_vector_to_pydf(py, feature_series)?)?;
            offset += chunk_length;
        }
        Ok(feature_dict.to_object(py))
    }

    fn insert(&mut self, table_name: String, data: &PyAny) -> PyResult<()> {
        let pkey: Vec<String> = self.connection.lrange(format!("{table_name}_pkey"), 0, -1).map_err(
            |e| PyConnectionError::new_err(e.to_string())
        )?;
        let columns = data.getattr("columns")?.cast_as::<PyList>()?;
        let mut rust_series: Vec<Series> = Vec::new();
        for col in columns {
            rust_series.push(py_series_to_rust_series(data.get_item(col)?)?);
        }
        let rust_df = DataFrame::new(rust_series).map_err(
            |e| PyValueError::new_err(e.to_string())
        )?;

        let key_expr = pkey.iter().map(
            |key| concat_str([lit(key.clone()), col(key).cast(DataType::Utf8)], "=")
        ).collect::<Vec<Expr>>();
        
        let keys_df = rust_df.clone().lazy().select([
            concat_str(key_expr, ";").alias("table_key")
        ]).collect().map_err(|e| PyValueError::new_err(e.to_string()))?;

        let keys = keys_df.column("table_key").unwrap().rechunk();
        let mut features = rust_df.lazy().select([col("*").exclude(pkey)]).collect().unwrap();
        features.rechunk();
        
        let mut pipe = redis::pipe();
        for column_name in features.get_column_names() {
            println!("Inserting {} feature", column_name);
            let column_values = features.column(column_name).unwrap();
            for (key, col_value) in keys.iter().zip(column_values.iter()) {
                let key_str = match key {
                    AnyValue::Utf8(val) => val,
                    _ => panic!("Invalid key type")
                };
                match col_value {
                    AnyValue::Float32(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Float64(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Int8(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Int16(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Int32(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Int64(val) => {pipe.hset::<_, _, _>(key_str, column_name, val);},
                    AnyValue::Null => (),
                    _ => panic!("wrong data type")
                }
            }
        }
        pipe.execute(&mut self.connection);
        Ok(())
    }
 }


#[pymodule]
fn feature_store(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<FeatureStoreClient>()?;
    Ok(())
}