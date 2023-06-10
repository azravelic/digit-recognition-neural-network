% function for digit display
function [d_image] = displayDigit(digit)
  colormap(gray);
  max_v = max(abs(digit));
  digit_vis = reshape(digit, 20, 20) / max_v;
  d_image = imagesc(digit_vis, [-1 1]);
end
