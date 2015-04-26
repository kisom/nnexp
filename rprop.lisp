(in-package :nnexp)

(defun matrix-gradients (mat &key (fn #'sigmoid-gradient))
  (let ((gmat nil))
    (lm:do-each-matrix-element (n mat)
      (push (funcall fn n) gmat))
    (lm:make-matrix (lm:matrix-rows mat)
		    (lm:matrix-cols mat)
		    :initial-elements gmat)))

(defun gradient-sign-change (layer &optional (update t))
  "Compute the gradient for the layer, returning a matrix of sign
changes and updating the layer with the current gradient. After a call
to this function, the layer is modified."
  (let* ((gradients   (matrix-gradients (weights-of layer)))
	 (sign-change (do-matrix
			  (do-matrix2 gradients
			    (gradients-of layer)
			    #'*)
			(kutils:compose #'neg #'sign))))
    (when update
      (setf (gradients-of layer) gradients))
    sign-change))

(defun Δw0 (layer)
  (labels ((fn (g Δ)
	     (* (neg (sign g)) Δ)))
    (let ((Δij (do-matrix2 (gradients-of layer) (Δ-of layer) #'fn)))
      (setf (Δ-of layer) Δij)
      (setf (weights-of layer)
	    (do-matrix2 Δij (weights-of layer) #'+)))))

(defun Δw- (layer)
  (setf (Δ-of layer)
	(do-matrix (Δ-of layer)
	  (lambda (n) (max (* n +NEGATIVE-Η+) +Δ-MINIMUM+)))))

(defun Δw+ (layer)
  (labels ((min-Η (n)
	     (min (* n +POSITIVE-Η+) +Δ-MAXIMUM+))
	   (fn (x y)
	     (* (neg (sign x)) y)))
    (let* ((Δij (do-matrix (Δ-of layer) #'min-Η))
	   (Δw  (do-matrix2 (gradients-of layer) Δij #'fn)))
      (setf (Δ-of layer) Δij)
      (setf (weights-of layer)
	    (do-matrix2 (weights-of layer) Δw #'+)))))

(defun reduce-vector (v fn &key (initial-value 0))
  (let ((result initial-value))
    (dotimes (i (lm:dimension v))
      (setf result (funcall fn result (lm:elt v i))))
    result))

(defun rprop (network results expected)
  (let ((ess (reduce-vector (lm:- (first results) (lm:to-vector expected))
			    (lambda (acc n) (+ acc (* n n))))))
    
    ess))
