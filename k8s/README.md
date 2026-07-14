# Kubernetes Deployment for SALMONN SQA API

This directory contains Kubernetes manifests for deploying the SALMONN SQA API to Azure Kubernetes Service (AKS) or any Kubernetes cluster with GPU support.

## Prerequisites

1. Kubernetes cluster with GPU nodes (NVIDIA GPU Operator installed)
2. kubectl configured to access your cluster
3. Docker image pushed to a container registry (ACR, Docker Hub, etc.)
4. Models uploaded to Azure Files or another persistent storage

## Files

- `deployment.yaml` - Main application deployment with GPU resources
- `service.yaml` - LoadBalancer and ClusterIP services
- `pvc.yaml` - Persistent Volume Claims for models and checkpoints
- `configmap.yaml` - Configuration for SALMONN inference
- `hpa.yaml` - Horizontal Pod Autoscaler for scaling based on load

## Deployment Steps

### 1. Upload Models to Azure Files

```bash
# Create storage account and file share
az storage account create --name salmonnstorage --resource-group salmonn-rg
az storage share create --name models --account-name salmonnstorage
az storage share create --name checkpoints --account-name salmonnstorage

# Upload models
az storage file upload-batch \
  --destination models \
  --source ./salmonn_sqa/models \
  --account-name salmonnstorage

az storage file upload-batch \
  --destination checkpoints \
  --source ./salmonn_sqa/ckpt \
  --account-name salmonnstorage
```

### 2. Create Secret for Storage Access

```bash
STORAGE_KEY=$(az storage account keys list \
  --account-name salmonnstorage \
  --query [0].value -o tsv)

kubectl create secret generic azure-storage-secret \
  --from-literal=azurestorageaccountname=salmonnstorage \
  --from-literal=azurestorageaccountkey=$STORAGE_KEY
```

### 3. Update PVC to Use Azure Files

Edit `pvc.yaml` to use Azure Files storage class:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: salmonn-models-pv
spec:
  capacity:
    storage: 20Gi
  accessModes:
    - ReadOnlyMany
  storageClassName: ""
  azureFile:
    secretName: azure-storage-secret
    shareName: models
    readOnly: true
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: salmonn-models-pvc
spec:
  accessModes:
    - ReadOnlyMany
  storageClassName: ""
  resources:
    requests:
      storage: 20Gi
  volumeName: salmonn-models-pv
```

### 4. Apply Manifests

```bash
# Apply in order
kubectl apply -f configmap.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f hpa.yaml
```

### 5. Verify Deployment

```bash
# Check pods
kubectl get pods -l app=salmonn-sqa-api

# Check logs
kubectl logs -l app=salmonn-sqa-api -f

# Check service
kubectl get svc salmonn-sqa-api

# Get external IP
kubectl get svc salmonn-sqa-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

### 6. Test API

```bash
EXTERNAL_IP=$(kubectl get svc salmonn-sqa-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Health check
curl http://$EXTERNAL_IP/health

# Test assessment
curl -X POST "http://$EXTERNAL_IP/assess" \
  -F "file=@/path/to/audio.wav"
```

## Scaling

The deployment includes an HPA that automatically scales based on CPU and memory usage.

**Manual scaling:**
```bash
kubectl scale deployment salmonn-sqa-api --replicas=3
```

**Check autoscaling status:**
```bash
kubectl get hpa salmonn-sqa-api
```

## Monitoring

**View logs:**
```bash
kubectl logs -l app=salmonn-sqa-api --tail=100 -f
```

**Check resource usage:**
```bash
kubectl top pods -l app=salmonn-sqa-api
```

**GPU utilization (if nvidia-smi is available):**
```bash
kubectl exec -it <pod-name> -- nvidia-smi
```

## Troubleshooting

**Pod not starting:**
```bash
kubectl describe pod -l app=salmonn-sqa-api
kubectl logs -l app=salmonn-sqa-api
```

**GPU not detected:**
- Ensure NVIDIA GPU Operator is installed: `kubectl get pods -n gpu-operator-resources`
- Verify node has GPU: `kubectl get nodes -o json | jq '.items[].status.allocatable'`

**Models not loading:**
- Check PVC is bound: `kubectl get pvc`
- Verify storage secret: `kubectl get secret azure-storage-secret`
- Check if files exist in volume: `kubectl exec -it <pod-name> -- ls -lh /app/salmonn_sqa/models`

## Cost Optimization

1. **Use node pools with autoscaling:**
   ```bash
   az aks nodepool update \
     --cluster-name salmonn-aks \
     --resource-group salmonn-rg \
     --name gpupool \
     --enable-cluster-autoscaler \
     --min-count 0 \
     --max-count 3
   ```

2. **Use Spot instances:**
   ```bash
   az aks nodepool add \
     --cluster-name salmonn-aks \
     --resource-group salmonn-rg \
     --name gpuspotpool \
     --node-vm-size Standard_NC6s_v3 \
     --priority Spot \
     --eviction-policy Delete \
     --spot-max-price -1 \
     --enable-cluster-autoscaler \
     --min-count 0 \
     --max-count 3
   ```

3. **Set pod disruption budget:**
   ```yaml
   apiVersion: policy/v1
   kind: PodDisruptionBudget
   metadata:
     name: salmonn-pdb
   spec:
     minAvailable: 1
     selector:
       matchLabels:
         app: salmonn-sqa-api
   ```

## Cleanup

```bash
kubectl delete -f hpa.yaml
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete -f pvc.yaml
kubectl delete -f configmap.yaml
kubectl delete secret azure-storage-secret
```
