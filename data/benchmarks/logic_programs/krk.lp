VAR king1_rank[A;B] 0 1
VAR king1_file[A;C] 0 1
VAR rook_rank[A;D] 0 1
VAR rook_file[A;E] 0 1
VAR king2_rank[A;F] 0 1
VAR king2_file[A;E] 0 1
VAR lt[F;D] 0 1
VAR lt[F;B] 0 1
VAR lt[D;B] 0 1
VAR lt[E;C] 0 1
VAR adj[F;F] 0 1
VAR adj[D;D] 0 1
VAR adj[B;B] 0 1
VAR adj[E;E] 0 1
VAR adj[C;C] 0 1
VAR king2_rank[A;B] 0 1
VAR lt[B;D] 0 1
VAR lt[C;E] 0 1
VAR king2_file[A;F] 0 1
VAR lt[E;F] 0 1
VAR lt[C;F] 0 1
VAR adj[F;C] 0 1
VAR adj[C;F] 0 1
VAR adj[D;B] 0 1
VAR adj[B;D] 0 1
VAR adj[E;C] 0 1
VAR adj[C;E] 0 1
VAR king2_file[A;C] 0 1
VAR king2_rank[A;D] 0 1
VAR lt[F;E] 0 1
VAR lt[F;C] 0 1
VAR rook_rank[A;B] 0 1
VAR rook_file[A;C] 0 1
VAR lt[D;F] 0 1
VAR adj[F;D] 0 1
VAR adj[D;F] 0 1
VAR lt[B;F] 0 1
VAR adj[F;B] 0 1
VAR adj[B;F] 0 1
VAR rook_file[A;D] 0 1
VAR king2_rank[A;E] 0 1
VAR lt[E;B] 0 1
VAR lt[C;D] 0 1
VAR adj[E;B] 0 1
VAR adj[B;E] 0 1
VAR king2_file[A;G] 0 1
VAR lt[E;G] 0 1
VAR lt[C;G] 0 1
VAR adj[G;G] 0 1
VAR adj[G;C] 0 1
VAR adj[C;G] 0 1
VAR adj[F;E] 0 1
VAR adj[E;F] 0 1
VAR lt[G;C] 0 1
VAR lt[D;C] 0 1
VAR lt[D;E] 0 1
VAR lt[G;E] 0 1
VAR adj[G;E] 0 1
VAR adj[E;G] 0 1
VAR lt[E;D] 0 1
VAR adj[D;C] 0 1
VAR adj[C;D] 0 1
VAR lt[B;E] 0 1
VAR king2_file[A;D] 0 1
VAR adj[E;D] 0 1
VAR adj[D;E] 0 1
VAR illegal[A] 0 1

illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;F](1,T-1), king2_file[A;G](1,T-1), adj[F;B](1,T-1), adj[G;C](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;E](1,T-1), king2_file[A;F](1,T-1), adj[E;B](1,T-1), adj[F;C](1,T-1).

illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), rook_rank[A;D](1,T-1), rook_file[A;E](1,T-1), king2_rank[A;D](1,T-1). 
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), rook_rank[A;D](1,T-1), rook_file[A;E](1,T-1), king2_file[A;E](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;B](1,T-1), king2_file[A;C](1,T-1).

illegal[A](1,T) :- rook_rank[A;D](1,T-1), rook_file[A;E](1,T-1), king2_rank[A;D](1,T-1), king2_file[A;E](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), rook_rank[A;B](1,T-1), rook_file[A;C](1,T-1).

illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;F](1,T-1), king2_file[A;C](1,T-1), adj[B;F](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;E](1,T-1), king2_file[A;C](1,T-1), adj[B;E](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;D](1,T-1), king2_file[A;C](1,T-1), adj[B;D](1,T-1).

illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;B](1,T-1), king2_file[A;G](1,T-1), adj[C;G](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;B](1,T-1), king2_file[A;E](1,T-1), adj[C;E](1,T-1).
illegal[A](1,T) :- king1_rank[A;B](1,T-1), king1_file[A;C](1,T-1), king2_rank[A;B](1,T-1), king2_file[A;F](1,T-1), adj[C;F](1,T-1).