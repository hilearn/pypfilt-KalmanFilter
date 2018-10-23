model RandomWalk {
  param q, r;
  noise eta;
  state v;
  obs rets;

  sub parameter {
    q <- 0.2;
    r <- 0.1;
  }

  sub proposal_parameter {
    q ~ log_gaussian(q, 1);
    r ~ log_gaussian(r, 1);
  }

  sub initial {
    v ~ gaussian(0, 0.2);
  }

  sub transition {
    eta ~ gaussian(0, q);
    v <- v + eta;
  }

  sub observation {
    rets ~ gaussian(v, r);
  }
}
