# build dataset according to given 'dataset_file'
def build_dataset(args):
    if args.dataset_file == 'shanghaitech_part_A':
        from crowd_datasets.SHHA.loading_data import loading_data
        return loading_data

    return None