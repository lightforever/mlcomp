import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";

@Injectable({
  providedIn: 'root'
})
export class DagService extends BaseService{
   protected collection_part: string = 'dags';
   protected single_part: string = 'dag';


}
