;;;; nnexp.lisp

(in-package #:nnexp)

;;; "nnexp" goes here. Hacks and glory await!
;;;
;;; The basic neural network should model a three-layer perceptron
;;; network; batched RPROP will be used to train the network.

(defconstant +DEFAULT-LIMIT+ 1.0)
(defconstant +ZERO-TOLERANCE+ 1.e-16)
(defconstant +Δ-MINIMUM+ 1e-6)
(defconstant +Δ-MAXIMUM+ 50)
(defconstant +NEGATIVE-Η+ 0.5)
(defconstant +POSITIVE-Η+ 1.2)
(defconstant +Δ-INITIAL+ 0.1)

(defclass layer ()
  ((bias      :initarg :bias      :accessor bias-of)
   (weights   :initarg :weights   :accessor weights-of)
   (gradients :initarg :gradients :accessor gradients-of)
   (Δ-update  :initarg :Δ         :accessor Δ-of))
  (:documentation "A layer contains the weights and bias for a layer in the neural network."))

(defun weight-random (limit)
  (/ (random limit) (random limit)))

(defun random-vector (n &key (limit +DEFAULT-LIMIT+) (rand #'random))
  "Create a random n-dimension vector."
  (let ((tmplst nil))
    (dotimes (i n)
      (push (funcall rand limit) tmplst))
    (lm:make-vector n :initial-elements tmplst)))

(defun random-matrix (m n &key (limit +DEFAULT-LIMIT+) (rand #'random))
  (let ((tmplst nil))
    (dotimes (j m)
      (dotimes (i n)
	(push (funcall rand limit) tmplst)))
    (lm:make-matrix m n :initial-elements tmplst)))

(defmacro do-vector (vec fn)
  (let ((tmp-vec (gensym)))
    `(let ((,tmp-vec nil))
       (lm:do-each-vector-element (n ,vec :index-symbol k)
	 (push (funcall ,fn n) ,tmp-vec))
       (lm:make-vector (lm:dimension ,vec)
		       :initial-elements (reverse,tmp-vec)))))

(defmacro do-matrix (mat fn)
  "Apply fn to each element in the matrix. The function"
  (let ((tmp-mat (gensym)))
    `(let ((,tmp-mat nil))
       (lm:do-each-matrix-element (n ,mat)
	 (push (funcall ,fn n) ,tmp-mat))
       (lm:make-matrix (lm:matrix-rows ,mat)
		       (lm:matrix-cols ,mat)
		       :initial-elements (reverse,tmp-mat)))))

(defmacro do-matrix2 (mat1 mat2 fn)
  "Apply fn to each element in both matrices. The function must accept
two arguments of the same type as the matrix elements, and it will
receive elements of the matrices in the same order they are provided."
  (let ((tmp-mat (gensym)))
    `(let ((,tmp-mat nil))
       (lm:do-each-matrix-element (n ,mat1 i j)
	 (push (funcall ,fn n (lm:matrix-elt ,mat2 i j)) ,tmp-mat))
       (lm:make-matrix (lm:matrix-rows ,mat1)
		       (lm:matrix-cols ,mat1)
		       :initial-elements (reverse,tmp-mat)))))

(defun Δ0-vector (k)
  (let ((v nil))
    (dotimes (i k)
      (push +Δ-INITIAL+ v))
    (lm:to-vector v)))

(defun Δ0-matrix (m n)
  (let ((v nil))
    (dotimes (i (* m n))
      (push +Δ-INITIAL+ v))
    (lm:make-matrix m n :initial-elements v)))

(defun make-layer (inputs neurons &key (limit +DEFAULT-LIMIT+))
  (make-instance 'layer
		 :weights   (random-matrix inputs neurons
					   :limit limit
					   :rand #'weight-random)
		 :bias      (random-vector neurons
					   :limit limit)
		 :gradients (lm:make-matrix inputs neurons)))

(defun multiple-p (lst)
  (when (listp lst)
   (cond 
      ((null lst) nil)
      ((null (cdr lst)) nil)
      (t t))))

(defun make-dimensions (dim &key (limit +DEFAULT-LIMIT+))
  (when (multiple-p dim)
    (cons (make-layer (first dim) (second dim) :limit limit)
	  (make-dimensions (rest dim)))))

(defun make-network-3 (inputs hidden outputs &key (limit +DEFAULT-LIMIT+))
  "Create a 3-layer network. The three input elements indicate the
number of input, hidden, and output neurons. For example, a 2-input,
3-hidden neuron, single output network (e.g. a XOR test network) would be
called with (MAKE-NETWORK-3 2 3 1)."
  (acons :layers
	 (make-dimensions (list inputs hidden outputs) :limit limit)
	 (acons :activation :sigmoid nil)))

(defun make-network (dim &key (limit +DEFAULT-LIMIT+))
  "Create a mulitple-dimension network. The dimensions should be
provided as a list where each element is the number of neurons in that
layer. For example, the two-input, single three-neuron hidden layer,
single output neuron would be built with (MAKE-NETWORK '(2 3 1))."
  (acons :layers
	 (make-dimensions dim :limit limit)
	 (acons :activation :sigmoid nil)))

(defun forward-inputs (imat layer)
  (lm:-
   (lm:* imat (weights-of layer))
   (bias-of layer)))

(defun neg (n)
  (- 0 n))

(defun sigmoid (n)
  (/ 1.0
     (+ 1.0
	(exp (neg n)))))

(defun sigmoid-gradient (n)
  (* n (- 1.0 n)))

(defun float-zerop (n)
  (< (abs n) +ZERO-TOLERANCE+))

(defun sign (n)
  (cond
    ((float-zerop n) 0)
    ((< n 0)        -1)
    ((> n 0)         1)))

(defun activate-layer (imat layer &key (fn #'sigmoid))
  (do-vector (forward-inputs imat layer) fn))

(defun activate-network (network inputs)
  "Forward the inputs through the network. The inputs should be a
list; the function will output a list of neuron layers where the first
element is the network's output and the last element is the input. All
the neuron layers are returned as L-MATH vectors suitable for passing
to the training function."
  (let ((neuron-layers (list (lm:to-vector inputs))))
    (dolist (layer (cdr (assoc :layers network)))
      (push (activate-layer (car neuron-layers) layer) neuron-layers))
    neuron-layers))

(defun describe-network (net)
  (dolist (layer (cdr (assoc :layers net)))
    (describe layer)
    (format t "============================~%")))


