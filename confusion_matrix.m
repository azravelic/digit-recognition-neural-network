%cnfmat
function cnf = confusion_matrix(y_pred, y)
  cnf = zeros(10,10);
  for i = 1:length(y_pred)
    pred_val = y_pred(i);
    act_val = y(i);
    if y_pred(i) == y(i)
      cnf(pred_val,pred_val) += 1;
    else
      cnf(pred_val,act_val) += 1;
    end
  end
end

