def SQuant_func(
    flip_number,
    flip_up_number,
    flip_down_number,
    
    rounding_error_sum,
    rounding_number, 
    rounding_error, 

    up_number, 
    up_error, 
    up_priority, 
    up_order, 

    down_number, 
    down_error, 
    down_priority,
    down_order,
):
    conver_shape = rounding_number.shape
    for oc in range(conver_shape[0]):
        for ic in range(conver_shape[1]):
            round_func(
                flip_number,
                flip_up_number,
                flip_down_number,
                
                rounding_error_sum[oc][ic],
                rounding_number[oc][ic], 
                rounding_error[oc][ic], 

                up_number[oc][ic], 
                up_error[oc][ic], 
                up_priority[oc][ic], 
                up_order[oc][ic], 

                down_number[oc][ic], 
                down_error[oc][ic], 
                down_priority[oc][ic],
                down_order[oc][ic], 
            )

def round_func(
    flip_number,
    flip_up_number,
    flip_down_number,
    rounding_error_sum,
    rounding_number_, 
    rounding_error_, 
    up_number_, 
    up_error_, 
    up_priority_,
    up_order_, 
    down_number_,
    down_error_, 
    down_priority_,
    down_order_,
):
    if rounding_error_sum < 0:
        # print("UP")
        number_ = up_number_
        error_ = up_error_
        priority_ = up_priority_
        order_  = up_order_
        error_1 = down_error_
        priority_1 = down_priority_
        flip_number_ = flip_up_number
        is_up = True
    else:
        # print("Down")
        number_ = down_number_
        error_ = down_error_
        priority_ = down_priority_
        order_  = down_order_
        error_1 = up_error_
        priority_1 = up_priority_
        flip_number_ = flip_down_number
        is_up = False

    rounding_error_sum = rounding_error_sum.abs()
    topk = int(rounding_error_sum.round().item())
    over_squant = (topk >= rounding_error_sum)

    idx_ = order_[0:topk]
    rounding_error_[idx_] =  error_[idx_]
    rounding_number_[idx_] = number_[idx_]

    if over_squant:
        idx_c = order_[topk - 1]
        priority_1[idx_c] = rounding_error_[idx_c].abs()
    else:
        idx_c = order_[topk]
        priority_[idx_c] = rounding_error_[idx_c].abs()