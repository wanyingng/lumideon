# ─── Outputs ──────────────────────────────────────────────────────────────────

output "alb_dns" {
  value = aws_lb.main.dns_name
}

output "app_ecr_url" {
  value = aws_ecr_repository.app.repository_url
}

output "pyrit_ecr_url" {
  value = aws_ecr_repository.pyrit.repository_url
}

output "tensorzero_ecr_url" {
  value = aws_ecr_repository.tensorzero.repository_url
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}

output "db_endpoint" {
  value     = aws_db_instance.postgres.endpoint
  sensitive = true
}
