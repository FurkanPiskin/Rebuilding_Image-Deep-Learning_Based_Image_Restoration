def log_results(loss_meter_dict):
    for loss_name, meter in loss_meter_dict.items():
        print(f"{loss_name}: {meter.avg:.6f}")

        