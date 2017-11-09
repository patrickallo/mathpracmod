((*- extends 'article.tplx' -*))
% Disable input cells
((* block input_group *))
((* endblock input_group *))

((* block bibliography *))
\bibliographystyle{unsrt}
\bibliography{ipython}
((* endblock bibliography *))
