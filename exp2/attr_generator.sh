source ./conf.sh

if [ -f "$dataset_attr_file" ]; then
    echo "$dataset_attr_file already exists."
    exit 1
fi


python ../tests/attr_generator.py --output_file ${dataset_attr_file} \
                            --N ${N} \
                            --attr_type_list ${attr_type} \
                            --categorical_attr_max_cardinality ${categorical_attr_max_cardinality} \
                            --numerical_max_attr ${numerical_max_attr}

