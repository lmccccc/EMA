
algo=Milvus_HNSW

vbase_schema_name="experiments"
vbase_table_name="${dataset}_${attr_index_type}"
vbase_dataset_file=$index_root"/"${vbase_table_name}"_data.csv"
echo "vbase schema name: "$vbase_schema_name
echo "vbase table name: "$vbase_table_name
echo "vbase dataset file: "$vbase_dataset_file