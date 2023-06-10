% "local" evaluation
function [l_accuracy, l_precision, l_recall, l_specificity, l_fscore] = l_evaluation(cnf_nos)
  l_accuracy = (cnf_nos(:,1) + cnf_nos(:,4))./(cnf_nos(:,1) + cnf_nos(:,2) + cnf_nos(:,3) + cnf_nos(:,4));
  l_precision = cnf_nos(:,1) ./ (cnf_nos(:,1) + cnf_nos(:,2));
  l_recall = cnf_nos(:,1) ./ (cnf_nos(:,1) + cnf_nos(:,3));
  l_specificity = cnf_nos(:,4) ./ (cnf_nos(:,4) + cnf_nos(:,2));
  l_fscore = (l_precision .* l_recall) ./ (l_precision + l_recall)
end

