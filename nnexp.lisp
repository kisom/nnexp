;;;; nnexp.lisp

(in-package #:nnexp)

;;; "nnexp" goes here. Hacks and glory await!
;;;
;;; The basic neural network should model a three-layer perceptron
;;; network; batched RPROP will be used to train the network.

(defconstant +DEFAULT-LIMIT+ 1.0)

(defclass layer ()
  ((bias    :initarg :bias :accessor bias-of)
   (weights :initarg :weights :accessor weights-of))
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

(defun make-layer (inputs neurons &key (limit +DEFAULT-LIMIT+))
  (make-instance 'layer
		 :weights (random-matrix inputs neurons :limit limit :rand #'weight-random)
		 :bias    (random-vector neurons :limit limit)))

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

(defun input-vector (inputs)
  "Convert a list of inputs to a matrix."
  (if (listp inputs)
      (lm:make-vector (length inputs) :initial-elements inputs)
      inputs))

(defun forward-inputs (imat layer)
  (lm:-
   (lm:* imat (weights-of layer))
   (bias-of layer)))

(defun sigmoid (n)
  (/ +DEFAULT-LIMIT+
     (+ +DEFAULT-LIMIT+
	(exp (- 0 n)))))

(defun activate-layer (imat layer &key (fn #'sigmoid))
  (let ((neurons (forward-inputs imat layer))
	(outputs nil))
    (dotimes (i (lm:dimension neurons))
      (push (funcall fn (lm:elt neurons i)) outputs))
    (lm:make-vector (lm:dimension neurons) :initial-elements outputs)))

(defun activate-network (network inputs)
  "Forward the inputs through the network. The inputs should be a
list; the function will output a list of neuron layers where the first
element is the network's output and the last element is the input. All
the neuron layers are returned as L-MATH vectors suitable for passing
to the training function."
  (let* ((imat          (input-vector inputs))
	 (neuron-layers (list imat)))
    (dolist (layer (cdr (assoc :layers network)))
      (describe layer)
      (push (activate-layer (car neuron-layers) layer) neuron-layers))
    neuron-layers))

(defun describe-network (net)
  (dolist (layer (cdr (assoc :layers net)))
    (describe layer)
    (format t "============================~%")))
