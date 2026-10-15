param location string = 'eastus'

resource sa 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'sample'
  location: location
}
