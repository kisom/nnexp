;;;; nnexp.lisp

(in-package #:nnexp)

;;; "nnexp" goes here. Hacks and glory await!
;;;
;;; The basic neural network should model a three-layer perceptron
;;; network; right now, online backpropagation is being used for
;;; training.

(defclass layer ()
  ((bias    :initarg :bias :accessor bias-of)
   (weights :initarg :weights :accessor weights-of))
  (:documentation "A layer contains the weights and bias for a layer in the neural network."))

(defun weight-random (limit)
  (/ (random limit) (random limit)))

(defun random-1 (n &key (limit 1.0) (rand #'random))
  "Create a random n-by-1 matrix."
  (let ((tmplst nil))
    (dotimes (i n)
      (push (funcall rand limit) tmplst))
    (lm:make-matrix n 1 :initial-elements tmplst)))

(defun random-matrix (m n &key (limit 1.0) (rand #'random))
  (let ((tmplst nil))
    (dotimes (j m)
      (dotimes (i n)
	(push (funcall rand limit) tmplst)))
    (lm:make-matrix m n :initial-elements tmplst)))

(defun make-layer (inputs neurons &key (limit 1.0))
  (make-instance 'layer
		 :weights (random-matrix inputs neurons :limit limit :rand #'weight-random)
		 :bias    (random-1      neurons :limit limit)))

(defun make-network-3 (inputs hidden outputs &key (limit 1.0))
  "Create a 3-layer network."
  (acons :layers
	 (list (make-layer inputs hidden  :limit limit)
	       (make-layer hidden outputs :limit limit))
	 (acons :activation :sigmoid nil)))

(defun input-matrix (inputs)
  "Convert a list of inputs to a matrix."
  (lm:make-matrix (length inputs) 1 :initial-elements inputs))

