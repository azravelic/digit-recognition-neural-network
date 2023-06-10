%accuracy
function acc = accuracy(y_pred, y_test)
  acc = sum(y_pred == y_test) / length(y_pred);
end
