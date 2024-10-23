# graphical utilities
.getParents <- function(amat, x) {
  unname(which(amat[x, ] == 1 & amat[, x] == 0))
}

getParents <- function(amat, x) {
  if (length(x) == 1) {
    .getParents(amat, x)
  } else {
    p <- lapply(x, function(.x) .getParents(amat, .x))
    p <- unique(unlist(p))
    setdiff(p, x)
  }
}

# possible ancestors -------
# possible ancestors of node x (|x| = 1)
# y: nodes through which path cannot go
.getPossAn <- function(amat, x, y = NULL) {
  m <- amat
  q <- v <- previous <- rep(0, length(m[, 1]))
  i <- k <- 1
  if (length(x) > 1) {
    stop("Need to do this node by node!\n")
  }
  q <- sort(x)
  tmp <- m
  previous[1] <- q[1]
  l <- 0
  counter1 <- counter2 <- 0
  while (q[k] != 0 & k <= i) {
    t <- q[k]
    v[k] <- t
    k <- k + 1
    if (counter2 == 0) {
      l <- l + 1
      counter2 <- counter1
      counter1 <- 0
    } else {
      counter2 <- counter2 - 1
    }
    for (j in 1:length(tmp[1, ])) if (tmp[t, j] != 0) {
      if ((tmp[j, t] == 0 & tmp[t, j] == 1) | (previous[k - 1] == t
          ) | (tmp[j, previous[k - 1]] == 0 & tmp[previous[k - 1], j] == 0))
        if (!(j %in% q) & !(j %in% y)) {
          i <- i + 1
          previous[i] <- t
          q[i] <- j
          counter1 <- counter1 + 1
        }
    }
  }
  v <- setdiff(v, c(0))
  return(v)
}

# a wrapper around that supports multivariate x
getPossAn <- function(amat, x, y = NULL) {
  if (length(x) == 1) {
    return(.getPossAn(amat, x, y = y))
  } else {
    an.x <- lapply(x, function(.x) .getPossAn(amat, .x, y = y))
    return(sort(unique(unlist(an.x))))
  }
}


# possible descendants --------
# get possible descendants of x (|x| = 1)
# y: nodes through which path cannot go
getPossDe <- function(amat, x, y = NULL) {
  m <- amat
  distance <- rep(NA, length(m[, 1]))
  q <- v <- previous <- rep(0, length(m[, 1]))
  i <- k <- 1
  if (length(x) > 1) {
    stop("Need to do this node by node!\n")
  }
  q <- sort(x)
  tmp <- m
  previous[1] <- q[1]
  l <- 0
  distance[1] <- l
  counter1 <- counter2 <- 0
  while (q[k] != 0 & k <= i) {
    t <- q[k]
    v[k] <- t
    k <- k + 1
    if (counter2 == 0) {
      l <- l + 1
      counter2 <- counter1
      counter1 <- 0
    } else {
      counter2 <- counter2 - 1
    }
    for (j in 1:length(tmp[1, ])) if (tmp[j, t] != 0) {
      if ((tmp[t, j] == 0 & tmp[j, t] == 1) | (previous[k - 1] == t
          ) | (tmp[j, previous[k - 1]] == 0 & tmp[previous[k - 1], j] == 0))
        if (!(j %in% q) & !(j %in% y)) {
          i <- i + 1
          previous[i] <- t
          q[i] <- j
          distance[i] <- l
          counter1 <- counter1 + 1
        }
    }

  }
  v <- setdiff(v, c(0))
  return(v)
}


# bucket decomposition -------
getBucketDecomp <- function(x, y, amat) {
  stopifnot(pcalg::isValidGraph(amat, type = "pdag"))
  # get undirected graph
  amat.un <- (amat + t(amat))/2
  amat.un[amat.un == 0.5] <- 0
  U <- igraph::graph.adjacency(amat.un, mode = "undirected")
  U <- igraph::as_graphnel(U)
  connected.comps <- RBGL::connectedComp(U)
  # get ancestors of y in the subgraph where x is removed
  amat.x.removed <- amat
  amat.x.removed[x, ] <- 0
  amat.x.removed[, x] <- 0
  an.y <- getPossAn(amat.x.removed, y)
  # get induced bucket partitioning of an.y
  induced.buckets <- lapply(connected.comps, function(C) intersect(as.integer(C), an.y))
  induced.buckets <- induced.buckets[lapply(induced.buckets, length) > 0]
  return(induced.buckets)
}
