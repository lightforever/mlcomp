import { Injectable } from '@angular/core';
import {BaseService} from "./base.service";

@Injectable({
  providedIn: 'root'
})
export class LogService extends BaseService{
   protected collection_part: string = 'logs';
   protected single_part: string = 'log';


}
