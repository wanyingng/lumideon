# ─── Variables ───────────────────────────────────────────────────────────────

variable "aws_region" {
  default = "ap-southeast-1"
}

variable "project" {
  default = "lumideon"
}

variable "app_image" {
  description = "ECR image URI for the main app"
}

variable "pyrit_image" {
  description = "ECR image URI for the PyRIT dashboard"
}

variable "tensorzero_image" {
  description = "ECR image URI for TensorZero gateway sidecar"
  default     = "placeholder"
}

variable "api_key" {
  description = "API key for authenticating requests to lumideon"
  sensitive   = true
  default     = ""
}

# Desired number of ECS task instances for the application
variable "app_desired_count" {
  description = "Initial number of app ECS tasks"
  type        = number
  default     = 1
}

# Minimum threshold of concurrent ECS tasks for application auto-scaling
variable "app_min_capacity" {
  description = "Minimum number of app ECS tasks for auto-scaling"
  type        = number
  default     = 1
}

# Maximum threshold of concurrent ECS tasks for application auto-scaling
variable "app_max_capacity" {
  description = "Maximum number of app ECS tasks for auto-scaling"
  type        = number
  default     = 5
}

variable "app_cpu" {
  description = "CPU units for app task (1024 = 1 vCPU)"
  default     = "2048"
}

variable "app_memory" {
  description = "Memory in MB for app task"
  default     = "4096"
}

variable "db_instance_class" {
  description = "RDS instance class"
  default     = "db.t3.micro"
}

variable "db_multi_az" {
  description = "Enable RDS Multi-AZ for high availability"
  default     = false
}

variable "redis_node_type" {
  description = "ElastiCache node type"
  default     = "cache.t3.micro"
}

variable "redis_num_cache_nodes" {
  description = "Number of Redis cache nodes"
  default     = 1
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  default     = 7
}

variable "cpu_scale_target" {
  description = "Target CPU utilization percentage for auto-scaling"
  default     = 70
}

# ACM SSL/TLS certificate ARN for HTTPS listener configuration.
# Leave empty when using default ALB domain or HTTP-only traffic.
variable "acm_certificate_arn" {
  description = "ACM certificate ARN for HTTPS. Leave empty to use HTTP only."
  type        = string
  default     = ""
}
